# -*- coding: utf-8 -*-
""" Catalog module for Level-2 and Level-2 product repositories"""

import os
import re
import sys
import fnmatch
import uuid
import time
import datetime
import dateutil
import tinydb
import numpy as np
from loguru import logger
from netCDF4 import Dataset, num2date

from dateperiods import DatePeriod
from pysiral.output import NCDateNumDef


class SIRALProductCatalog(object):
    """ Parent catalog class for product catalogs for different
    data processing levels """

    def __init__(self,
                 repo_path,
                 auto_id=True,
                 repo_id=None,
                 processing_level=None,
                 period_id_level=None):
        self.repo_path = repo_path
        self._repo_id = repo_id
        if repo_id is not None:
            auto_id = False
        self.auto_id = auto_id
        self.processing_level = processing_level
        self.period_id_level = period_id_level
        self._catalog = {}

    def run_checks(self, check_list, raise_on_failure=True):
        """Runs a list of built-in queries. Valid checks: `is_single_hemisphere`, `is_single_version`
        
        Arguments:
            check_list {str list} -- list of checks to run
        
        Keyword Arguments:
            raise_on_failure {bool} -- Raise an SystemExi exception if a check is negative (default: {True})
        
        Returns:
            [bool list] -- flag list with check results (True: check passed)
                           Only returned when `raise` keyword is set to `False`
        """

        check_passed = np.zeros(np.shape(check_list), dtype=bool)
        for index, check in enumerate(check_list):
            check_passed[index] = getattr(self, check)
            try: 
                check_passed[index] = getattr(self, check)
            except AttributeError:
                logger.error("invalid check: %s" % str(check))
                check_passed[index] = False
            finally:
                if raise_on_failure and not check_passed[index]:
                    logger.error("failed check: %s" % str(check))
                    sys.exit()

        return check_passed

    def limit_to_period(self, limit_tcs_dt, limit_tce_dt):
        """
        Removes all catalog entries that have no overlap with start and end times (not reversible)
        TODO: think of a way of making it reversible or return a catalog subset
        """

        # Search for products that have no overlap (including partial overall)
        product_ids_not_in_subset = [
            product.id
            for product in self.product_list
            if not product.has_overlap(limit_tcs_dt, limit_tce_dt)
        ]

        # Remove from Catalog
        msg = "Removed %g products from catalog" % len(product_ids_not_in_subset)
        logger.info(msg)
        for product_id in product_ids_not_in_subset:
            self._catalog.pop(product_id, None)

    def get_product_info(self, product_id):
        return self._catalog.get(product_id, None)

    def append(self, ctlg, duplication=False):
        """Add the information from another catalog with rules for handling of duplicates 
        (same period, same mission)
        
        Arguments:
            ctlg {object} -- of type SIRALProductCatalog
        
        Keyword Arguments:
            duplication {bool} -- Allows (True) / Prevents (False) duplicate entries (same platform, same period)
                                in the merged catalog (default: {False})
        
        Returns:
            
        """

        # Perform Consistency checks
        # 1. argument need to be of type SIRALProductCatalog
        if not isinstance(ctlg, (L2IProductCatalog, L2PProductCatalog, L3CProductCatalog)):
            msg = "Invalid catalog (%s), must be from pysiral.catalog module"
            msg %= (str(ctlg.__class__.__name__))
            raise ValueError(msg)

        # 2. Catalogs need to be of the same processing level (l2p, l3c, ....)
        if self.processing_level != ctlg.processing_level:
            msg = "Invalid processing level (%s) of new catalog, %s required for appending"
            msg %= (str(ctlg.processing_level), str(self.processing_level))
            raise ValueError(msg)

        # Merge the catalogs
        for new_product in ctlg.product_list:

            # Check if new product is duplication in current catalog
            is_duplication = self._is_duplication(new_product)

            # if duplication ok & is duplication -> add
            if duplication and is_duplication or not is_duplication:
                self._catalog[new_product.id] = new_product

            else:
                continue

    def to_tinydb(self, filepath):
        """
        Write the content of the catalog to a tinydb json file

        :param filepath:

        :return:
        """
        db = tinydb.TinyDB(filepath)
        db.insert_multiple(item.tinydb_document for item in self.product_list)

    def query_datetime(self, dt, return_value="bool"):
        """ Searches the repository for products for a given datetime
        
        Arguments:
            dt {datetime} -- datetime definition for the query
        
        Keyword Arguments:
            return_value {str} -- Defines the type of output: `bool` for True/False flag, `products` for
                                  product path (list) and id for ids (default: {"bool"})
        
        Returns:
            [bool or str] -- see keyword `return`
        """

        if not isinstance(dt, datetime.datetime):
            raise ValueError("Argument dt needs to be datetime (was: %s)" % (type(dt)))

        product_ids = [prd.id for prd in self.product_list if prd.has_coverage(dt)]
        product_path = [prd.path for prd in self.product_list if prd.has_coverage(dt)]

        if return_value == "ids":
            return product_ids
        elif return_value == "products":
            return product_path
        else:
            return len(product_ids) > 0

    def query_overlap(self, tcs, tce, return_value="path"):
        """ Searches the repository for products that have overlapping coverage
        with a given time range
        
        Arguments:
            tcs {datetime} -- time coverage start of search period
            tce {datetime} -- time coverage end of search period
        
        Keyword Arguments:
            return_value {str} -- Defines the type of output: `bool` for True/False flag and `products` for
                                  product path (list) (default: {"bool"})
        
        Returns:
            [bool or str] -- see keyword `return`
        """

        if not isinstance(tcs, datetime.datetime):
            raise ValueError("Argument tcs needs to be datetime (was: %s)" % (type(tce)))

        if not isinstance(tce, datetime.datetime):
            raise ValueError("Argument tce needs to be datetime (was: %s)" % (type(tce)))

        # Search files
        product_ids = [prd.id for prd in self.product_list if prd.has_overlap(tcs, tce)]
        if return_value == "path":
            return [prd.path for prd in self.product_list if prd.has_overlap(tcs, tce)]
        elif return_value == "period_id":
            return [
                prd.period_id
                for prd in self.product_list
                if prd.has_overlap(tcs, tce)
            ]

        else:
            return product_ids

    def get_all_winter_ids(self, return_value="ids"):
        """ Returns a list of ids lists for each winter in the catalogue """

        # Get time range
        first_year, first_month = self.tcs.year, self.tcs.year
        last_year, last_month = self.tce.year, self.tce.year

        # Verify start year & end year
        # (must be that of winter beginning of October)
        if first_month < 10:
            first_year -= 1
        if last_month < 10:
            last_year -= 1

        winter_ids = []
        for winter_start_year in np.arange(first_year, last_year+1):
            time_range = self.get_winter_time_range(winter_start_year)
            ids = self.query_overlap(time_range.tcs.dt, time_range.tce.dt, return_value=return_value)
            winter_ids.append(ids)

        return winter_ids

    @staticmethod
    def get_winter_time_range(start_year):
        winter_start_tuple = [start_year, 10]
        winter_end_tuple = [start_year+1, 4]
        return DatePeriod(winter_start_tuple, winter_end_tuple)

    def get_northern_winter_netcdfs(self, start_year):
        """Returns a list for northern winter data for the period october through april
        
        Arguments:
            start_year {int} -- start year for the winter (yyyy-oct till yyyy+1-apr)

        Returns: 
            product_files {str list} -- list of files for specified winter season
        """

        # Construct time range 
        time_range = self.get_winter_time_range(start_year)

        # Query time range
        product_files = self.query_overlap(time_range.tcs.dt, time_range.tce.dt)

        # Reporting
        msg = "Found %g %s files for winter season %g/%g"
        msg %= (len(product_files), self.processing_level, start_year, start_year+1)
        logger.info(msg)

        return product_files

    def get_time_range_ids(self, tcs, tce):
        time_range = DatePeriod(tcs, tce)
        return self.query_overlap(
            time_range.tcs.dt, time_range.tce.dt, return_value="ids"
        )

    def get_month_products(self, month_num, exclude_years=None, platform="all"):
        """Returns a list all products that have coverage for a given month
        
        Arguments:
            month {int} -- month number (1-Jan, ..., 12:Dec)

        Returns: 
            product_files {tuple list} -- ((year, month), [list of files month])
        """

        # Query time range
        product_ids = []

        platforms = self.platforms if platform == "all" else [platform]
        years_to_include = [] if exclude_years is None else exclude_years
        n_products = 0
        for year in self.years:
            if year in years_to_include:
                continue
            time_range = DatePeriod([year, month_num], [year, month_num])
            tcs, tce = time_range.tcs.dt, time_range.tce.dt
            ids = [prd.id for prd in self.product_list if prd.has_overlap(tcs, tce) and prd.platform in platforms]
            n_products += len(ids)
            product_ids.extend(ids)

        # Reporting
        msg = "Found %g %s files for %s"
        month_name = datetime.datetime(2000, month_num, 1).strftime("%B")
        msg %= (n_products, self.processing_level, month_name)
        logger.info(msg)

        return product_ids

    def has_unique_doi(self, ref_doi):
        """ Returns True/False if all products have the reference doi """

        # 2 Step Procedure
        #  1. Test if DOI is unique (true means only one value, but could be None (default values))
        #  2. Test if unique doi is the reference doi

        dois_in_ctlg = self.dois_in_catalog
        prd_dois_are_unique = len(dois_in_ctlg) == 1

        # More than one doi in catalog -> test failed
        if not prd_dois_are_unique:
            return False

        return dois_in_ctlg[0] == ref_doi

    def _catalogize(self):
        """Create the product catalog of the repository"""

        # Get the list of netcdf product files
        nc_files = self.nc_files

        # Create the catalog entries as dictionary with product id as keys
        logger.info(
            'Catalogizing %s repository: %s (%g files)'
            % (self.processing_level, str(self.repo_path), len(nc_files))
        )

        if self.auto_id:
            subfolders = self.repo_path.split(os.sep)
            try:
                search = [bool(re.search(self.processing_level, subfolder)) for subfolder in subfolders]
                index = search.index(True)
                repo_id = subfolders[index-1]
            except IndexError:
                logger.warning("Auto id failed, did not find `%s` in list of subfolders" % self.processing_level)
                repo_id = None
            self._repo_id = repo_id
            logger.info("%s repository ID: %s" % (self.processing_level, str(self.repo_id)))

        nc_access_times = []
        t0 = time.process_time()
        for nc_file in self.nc_files:
            product_info = ProductMetadata(nc_file, target_processing_level=self.processing_level)
            nc_access_times.append(product_info.nc_access_time)
            self._catalog[product_info.id] = product_info
        t1 = time.process_time()
        if self.n_product_files > 0:
            logger.info("... done in %.1f seconds" % (t1-t0))
            logger.info("... average netCDF access time: %.4f sec" % np.mean(nc_access_times))

    def _is_duplication(self, product_info):
        """ Tests if specified product is a duplicate to the current catalog """
        return product_info.period_id in self.period_ids

    @property
    def nc_files(self):
        """Lists all netcdf files (*.nc) in the repository.
        
        Returns:
            [str] -- list of netcdf files
        """
        nc_files = []
        for root, dirnames, filenames in os.walk(self.repo_path):
            nc_files.extend(
                os.path.join(root, filename)
                for filename in fnmatch.filter(filenames, "*.nc")
            )

        return sorted(nc_files)

    @property
    def repo_id(self):
        return str(self._repo_id)

    @property
    def dois(self):
        return [prd.doi for prd in self.product_list]

    @property
    def dois_in_catalog(self):
        return np.unique(self.dois)

    @property
    def n_product_files(self):
        return len(self._catalog.keys())

    @property
    def product_ids(self):
        return sorted(self._catalog.keys())

    @property
    def duration(self):
        return list({item.time_coverage_duration for item in self._catalog.values()})

    @property
    def period_ids(self):
        return [self._catalog[idstr].period_id for idstr in self.product_ids]

    @property
    def product_list(self):
        return [self._catalog[idstr] for idstr in self.product_ids]

    @property
    def platforms(self):
        return np.unique([prd.platform for prd in self.product_list])

    @property
    def versions(self):
        return [prd.product_version for prd in self.product_list]

    @property
    def hemispheres(self):
        return [prd.hemisphere for prd in self.product_list]

    @property
    def hemisphere_list(self):
        return np.unique(self.hemispheres)

    @property
    def is_single_version(self):
        return len(np.unique(self.versions)) == 1

    @property
    def is_single_hemisphere(self):
        return len(self.hemisphere_list) == 1

    @property
    def is_north(self):
        hemisphere_list = self.hemisphere_list
        return self.is_single_hemisphere and hemisphere_list[0] == "north" 

    @property
    def is_south(self):
        hemisphere_list = self.hemisphere_list
        return self.is_single_hemisphere and hemisphere_list[0] == "south" 

    @property
    def years(self):
        years = sorted([prd.time_coverage_start.year for prd in self.product_list])
        return np.unique(years)

    @property
    def time_coverage_start(self):
        tcs = [prd.time_coverage_start for prd in self.product_list]
        return np.min(tcs)

    @property
    def tcs(self):
        """ Abbrevivation for self.time_coverage_start """
        return self.time_coverage_start

    @property
    def time_coverage_end(self):
        tce = [prd.time_coverage_end for prd in self.product_list]
        return np.max(tce)

    @property
    def tce(self):
        """ Abbrevivation for self.coverage_end """
        return self.time_coverage_end


class L2IProductCatalog(SIRALProductCatalog):
    """Catalog class for L3C product repositories

    Arguments:
            repo_path {str} -- path to repository"""

    def __init__(self, *args, **kwargs):
        kwargs.update(processing_level="l2i", period_id_level="daily")
        super(L2IProductCatalog, self).__init__(*args, **kwargs)
        self._catalogize()


class L2PProductCatalog(SIRALProductCatalog):
    """Catalog class for L2P product repositories

    Arguments:
            repo_path {str} -- path to repository"""

    def __init__(self, *args, **kwargs):
        kwargs.update(rocessing_level="l2p")
        super(L2PProductCatalog, self).__init__(*args, **kwargs)
        self._catalogize()


class L3CProductCatalog(SIRALProductCatalog):
    """Catalog class for L3C product repositories

    Arguments:
            repo_path {str} -- path to repository"""

    def __init__(self, *args, **kwargs):
        kwargs.update(processing_level="l3c", period_id_level="daily")
        super(L3CProductCatalog, self).__init__(*args, **kwargs)
        self._catalogize()


class L4ProductCatalog(SIRALProductCatalog):
    """Catalog class for L4 product repositories

    Arguments:
            repo_path {str} -- path to repository"""

    def __init__(self, *args, **kwargs):
        kwargs.update(processing_level="l4", period_id_level="daily")
        super(L4ProductCatalog, self).__init__(*args, **kwargs)
        self._catalogize()


class ProductMetadata(object):
    """Metadata data container for pysiral product files."""

    VALID_PROCESSING_LEVELS = ["l2i", "l2p", "l3c", "l4"]
    VALID_CDM_DATA_LEVEL = ["Trajectory", "Grid"]
    NC_PRODUCT_ATTRIBUTES = [
        "processing_level", "product_version", "cdm_data_level", "platform", 
        "time_coverage_start", "time_coverage_end", "product_timeliness",
        "time_coverage_duration", "source_mission_id", "source_hemisphere",
        "geospatial_lat_min", "geospatial_lat_max",
        "geospatial_lon_min", "geospatial_lon_max",
        "doi"]

    def __init__(self, path, target_processing_level=None):
        """
        Arguments:
            local_path {str} -- local path to the product netcdf
        
        Keyword Arguments:
            target_processing_level {str} -- Target processing level for the product netcdf. Settings 
                                             a specific processing level will cause an exception in 
                                             the case of a mismatch (default: {None})
        """
        self.path = path
        self._attr_dict = {}
        self.unique_str = str(uuid.uuid4())[:8]

        if target_processing_level in self.VALID_PROCESSING_LEVELS or target_processing_level is None:
            self._targ_proc_lvl = target_processing_level
        else:
            raise ValueError("Invalid target processing level: %s" % str(target_processing_level))

        # Fetch attributes (if possible) from netcdf
        t0 = time.process_time()
        nc = ReadNC(self.path, global_attrs_only=True)
        t1 = time.process_time()
        self.nc_access_time = t1-t0

        for attribute in self.NC_PRODUCT_ATTRIBUTES:

            # Extract value from netcdf global attributes
            try:
                value = getattr(nc, attribute)
            except AttributeError:
                value = None

            # Now follow a few special rules from some attributes
            if attribute == "processing_level":
                value = self._validate_proc_lvl(value)

            if attribute in ["time_coverage_start", "time_coverage_end"]:
                value = self._parse_datetime_definition(value)

            if re.search("geospatial", attribute):
                value = float(value)

            self._attr_dict[attribute] = value

        self._attr_dict["path"] = self.path

    def has_coverage(self, dt):
        """Test if datetime object is covered by product time coverage
        
        Arguments:
            dt {datetime} -- A datetime object that will be tested for coverage

        Returns:
            [bool] -- A True/False flag
        """
        return self.time_coverage_start <= dt <= self.time_coverage_end

    def has_overlap(self, tcs, tce):
        """Test if product has overlap with period defined by start & end datetime
        
        Arguments:
            tcs {datetime} -- A datetime object that defines start of time coverage test
            tce {datetime} -- A datetime object that defines end of time coverage test

        Returns:
            [bool] -- A True/False flag
        """

        # Validity check
        if tce <= tcs:
            raise ValueError("Invalid overlap test: tce <= tcs")

        no_coverage = tce <= self.time_coverage_start or tcs >= self.time_coverage_end
        return not no_coverage

    @staticmethod
    def _parse_datetime_definition(value):
        """Converts the string representation of a date & time into a
        datetime instance

        Arguments:
            value {str} -- [description]

        Returns:
            [datetime] -- [description]
        """
        d = dateutil.parser.parse(value)

        # Test if datetime is timezone aware
        # (true) -> remove time zone
        if d.tzinfo is not None and d.tzinfo.utcoffset(d) is not None:
            d = d.replace(tzinfo=None)

        return d

    def _validate_proc_lvl(self, value):
        """Validates the processing level str from the netcdf file: a) only save the id and 
        not any potential description
        
        Arguments:
            value {str} -- The attribute value from the product netcdf
        
        Raises:
            ValueError -- if mismatch between processing level in the product and the target level 
        
        Returns:
            [str] -- The validated processing level id string
        """

        # Make sure the value for processing level is only the id
        # search for the occurrence of all valid processing levels in the processing_level attribute
        pl_match = [pl in str(value).lower() for pl in self.VALID_PROCESSING_LEVELS]
        try: 
            index = pl_match.index(True)

        # NOTE: if the processing_level attribute does not exist, there is no choice but to trust the repo
        except ValueError:
            return self._targ_proc_lvl

        # Check if processing level in file matches target processing level
        value = self.VALID_PROCESSING_LEVELS[index]
        if value != self._targ_proc_lvl is not None:
            msg = "Invalid product processing level: %s (target: %s)"
            raise ValueError(msg % (value, self._targ_proc_lvl))

        return value

    def _get_datetime_label(self, dt):
        if self.processing_level in ["l2p", "l3c", "l3s"]:
            return dt.strftime("%Y%m%d")
        else:
            return dt.strftime("%Y%m%d%H%M%S")

    @property
    def id(self):
        """Generates an id string for the product.
        
        Returns:
            [str] -- id str of the product
        """
        
        identifier = (
            self.processing_level, str(self.platform),
            self.period_id, self.unique_str)
        return "%s-%s-%s-%s" % identifier

    @property
    def tcs(self):
        return self.time_coverage_start

    @property
    def tce(self):
        return self.time_coverage_end

    @property
    def tcs_label(self):
        return self._get_datetime_label(self.time_coverage_start)

    @property
    def tce_label(self):
        return self._get_datetime_label(self.time_coverage_end)

    @property
    def period_id(self):
        """ Generates a period id """
        identifier = (self.tcs_label, self.tce_label, self.time_coverage_duration)
        return "%sT%s-%s" % identifier

    @property
    def ref_time(self):
        tcs, tce = self.tcs, self.tce
        return tcs + (tce - tcs)/2

    @property
    def hemisphere(self):
        """ Determine the hemisphere based on the geospatial attributes """
        if self.geospatial_lat_min > 0.0:
            return "north"
        elif self.geospatial_lat_max < 0.0:
            return "south"
        else:
            return "global"

    @property
    def tinydb_document(self):
        """ Returns a tinydb document of this class """
        attr_dict = self._attr_dict.copy()
        for key, value in attr_dict.items():
            if isinstance(value, datetime.datetime):
                attr_dict[key] = value.isoformat()
        attr_dict["id"] = self.id
        return attr_dict


    def __getattr__(self, item):
        """
        Modify the attribute getter to provide a shortcut to the data content
        :param item: Name of the parameter
        :return:
        """
        if item in list(self._attr_dict.keys()):
            return self._attr_dict[item]
        else:
            raise AttributeError()


class ReadNC(object):
    """
    Quick & dirty method to parse content of netCDF file into a python object
    with attributes from file variables
    """
    def __init__(self, filename, verbose=False, autoscale=True,
                 nan_fill_value=False, global_attrs_only=False):
        self.keys = []
        self.time_def = NCDateNumDef()
        self.parameters = []
        self.attributes = []
        self.verbose = verbose
        self.autoscale = autoscale
        self.global_attrs_only = global_attrs_only
        self.nan_fill_value = nan_fill_value
        self.filename = filename
        self.parameters = []
        self.read_globals()
        self.read_content()

    def read_globals(self):
        pass

    def read_content(self):

        # Open the file
        try:
            f = Dataset(self.filename)
            f.set_auto_scale(self.autoscale)
        except RuntimeError as e:
            raise IOError("Cannot read netCDF file: %s" % self.filename) from e

        # Get the global attributes
        for attribute_name in f.ncattrs():

            self.attributes.append(attribute_name)
            attribute_value = getattr(f, attribute_name)

            # Convert timestamps back to datetime objects
            # TODO: This needs to be handled better
            if attribute_name in ["start_time", "stop_time"]:
                attribute_value = num2date(
                    attribute_value, self.time_def.units,
                    calendar=self.time_def.calendar)
            setattr(self, attribute_name, attribute_value)

        # Get the variables
        if not self.global_attrs_only:
            for key in f.variables.keys():

                variable = f.variables[key][:]

                try:
                    is_float = variable.dtype in ["float32", "float64"]
                    has_mask = hasattr(variable, "mask")
                except AttributeError:
                    is_float, has_mask = False, False

                if self.nan_fill_value and has_mask and is_float:
                    is_fill_value = np.where(variable.mask)
                    variable[is_fill_value] = np.nan

                setattr(self, key, variable)
                self.keys.append(key)
                self.parameters.append(key)
                if self.verbose:
                    print(key)
            self.parameters = f.variables.keys()
        f.close()
