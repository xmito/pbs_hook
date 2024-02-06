import os
import abc
import sys

# Force pbs_python to consider modules in other locations
if "/usr/local/lib/python3.6/site-packages" not in sys.path:
	sys.path.append("/usr/local/lib/python3.6/site-packages")
if "/usr/local/lib64/python3.6/site-packages" not in sys.path:
	sys.path.append("/usr/local/lib64/python3.6/site-packages")
if "/usr/lib/python3.6" not in sys.path:
	sys.path.append("/usr/lib/python3.6")
if "/usr/lib64/python3.6" not in sys.path:
	sys.path.append("/usr/lib64/python3.6")

import pbs
import math
import time
import builtins
import numpy as np
from psycopg2 import connect, sql, OperationalError
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

###############################################################################
################################## MODELS #####################################
###############################################################################

class Fallback(metaclass=abc.ABCMeta):
    req_cwalltime = True  # Whether the model requires current job walltime estimate
    req_hwalltime = True  # Whether the model requires historic jobs with walltime estimate
    entries = 0			 # Maximum number of required finished jobs
    @abc.abstractmethod
    def fit(self, walltime, run_time):
        return

    @abc.abstractmethod
    def predict(self, walltime):
        return
    
    def is_fallback(self):
        return True

class AvgTwo(Fallback):
    req_cwalltime = False
    req_hwalltime = False
    entries = 2
    def fit(self, walltime, run_time):
        self.result = np.average(run_time[:2])

    def predict(self, walltime):
        return self.result

class RelativeWalltimeUsage(Fallback):
    req_cwalltime = True
    req_hwalltime = True
    entries = 5
    def fit(self, walltime, run_time):
        self.wusage = np.max(run_time[:5] / walltime[:5])

    def predict(self, walltime):
        return self.wusage * walltime

class Walltime(Fallback):
    req_cwalltime = True
    req_hwalltime = False
    entries = 0
    def fit(self, walltime, run_time):
        pass

    def predict(self, walltime):
        return walltime

Fallback.register(AvgTwo)
Fallback.register(RelativeWalltimeUsage)
Fallback.register(Walltime)


class PolyRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, lrate, alpha=0.1):
        self.step = 1
        self.lrate = lrate
        self.fuzz = 10 ** -7
        self.alpha = alpha

    def _init_model(self, wide):
        self.weights = np.zeros(wide)
        # Stores magnitude of each feature
        # 10**-7 prevents 0 / 0 = Nan in algorithm
        self.mag = np.full((wide,), self.fuzz)

        # Cumulates gradient for each feature
        self.adapt_gd = np.zeros(wide)

        # Used to make the learning rate control the average change
        # in prediction from an update. N is initialized using fuzz
        # factor to prevent numeric problems in algorithm
        self.n = self.fuzz

    def _gradient(self, X, y):
        error = np.dot(X, self.weights) - y
        dreg = self.alpha * self.weights
        if error >= 0:
            return X + dreg
        else:
            return 2 * error * X + dreg

    def fit(self, X, y):
        X = np.array(X, ndmin=2)
        y = np.array(y, ndmin=1)
        X, y = check_X_y(X, y, y_numeric=True)
        if self.step == 1:
            self._init_model(X.shape[1])

        for xi, yi in zip(X, y):
            # Change feature magnitudes and weights
            chgmag = np.abs(xi) > self.mag
            self.weights[chgmag] *= self.mag[chgmag] / np.abs(xi[chgmag])
            self.mag[chgmag] = np.abs(xi[chgmag])
            self.n += np.sum(xi ** 2 / self.mag ** 2)

            grad = self._gradient(xi, yi)
            self.adapt_gd += grad ** 2
            self.weights -= (
                self.lrate
                * math.sqrt(self.step / self.n)
                * np.power(self.mag, -1)
                * np.power(np.sqrt(self.adapt_gd + self.fuzz), -1)
                * grad
            )
            self.step += 1
        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = np.array(X, ndmin=2)
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        return np.dot(X, self.weights)

    def get_params(self, deep=True):
        return {
            "lrate": self.lrate,
            "alpha": self.alpha,
        }

    def is_fallback(self):
        return False

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

#####################################################################################
################################ AUXILIARY FUNCTIONS ################################
#####################################################################################

def convert_size(value, units='b'):
    """Converts a string containing a size specification (e.g '1m')
       to a string using different units (e.g '1024k'). This function
       only interprets a decimal number at the start of the string,
       stopping at any unrecognized character and ignoring the rest
       of the string. When down-converting (e.g MB to KB), all calcu-
       lations involve integers and the result returned is exact. When
       up-converting (e.g KB to MB) floating point numbers are involved.
       The result is rounded up. For example:
       1023MB -> GB yields 1g
       1024MB -> GB yields 1g
       1025MB -> GB yields 1g
       Pattern matching or conversion may result in exceptions.
    """
    import re
    pbs.logmsg(pbs.EVENT_DEBUG3, f"Function 'convert size' called with the value: {value}, units: {units} ")
    logs = {'b': 0, 'kb': 10, 'mb': 20, 'gb': 30, 'tb': 40, 'pb': 50, 'eb': 60, 'zb': 70, 'yb': 80}
    try:
        new = units.lower()
        if new not in logs:
            new = 'b'
        val, old = re.match("([-+]?\d+)([bkmgtpezy]?b?)", str(value).lower()).groups()
        val = int(val)
        if val < 0:
            raise ValueError("Value may not be negative")
        if old not in logs.keys():
            old = 'b'
        factor = logs[old] - logs[new]
        val *= 2 ** factor
        slop = val - int(val)
        val = int(val)
        if slop > 0:
            val += 1
        # pbs.size() does not like units following zero
        if val <= 0:
            pbs.logmsg(pbs.EVENT_DEBUG3, f"Return value: 0")
            return '0'
        else:
            pbs.logmsg(pbs.EVENT_DEBUG3, "Return value: {str(val) + new}")
            return str(val) + new
    except:
        pbs.logmsg(pbs.EVENT_DEBUG3, "Return value: None")
        return None

def size_as_int(value, units='b'):
    """ Convert pbs.size value to integer representation in chosen units"""
    import string
    return int(convert_size(value, units).rstrip(string.ascii_lowercase))

def _reject(event, msg=None, trace_exc=False):
    if trace_exc:
        import traceback
        hook_event = [
            "queuejob",
            "modifyjob",
            "resvsub",
            "movejob",
            "runjob",
            "provision",
            "execjob_begin",
            "execjob_prologue",
            "execjob_epilogue",
            "execjob_end",
            "execjob_preterm",
            "execjob_launch",
            "exechost_periodic",
            "exechost_startup",
            "execjob_attach",
            "periodic",
            "resv_end"]

        exc_type, exc_value, exc_tb = sys.exc_info()
        trb = traceback.format_exception(exc_type, exc_value, exc_tb)
        trb[0] = f" Exception occured in the hook '{event.hook_name}'"\
             + f" during {hook_event[int(math.log2(event.type))]}\n"
        if msg:
            trb.append("Note: " + msg + '\n')
        msg = "".join(trb)
    event.reject(f"Hook {event.hook_name}: {msg}")

def _accept(event, msg=None, loglevel=pbs.LOG_DEBUG):
    if msg:
        pbs.logmsg(loglevel, "Hook %s: " % event.hook_name + msg)
    event.accept()


##########################################################################
################### HANDLER INTERFACE & IMPLEMENTATION ###################
##########################################################################

class DPInterface(metaclass=abc.ABCMeta):
    """ Data preprocessing interface declares abstract methods, that
       need to be implemented in order prepare data for model training
       and storing into database. This interface makes it easy to make
       changes to predictors set, because predictors are constructed 
       only in these methods"""

    _predictors = {"walltime": "INTEGER",
		  "ncpus": "INTEGER",
		  "mem": "BIGINT",
		  "last_1": "INTEGER",
		  "last_2": "INTEGER",
		  "last_3": "INTEGER",
		  "avg_2": "INTEGER",
		  "avg_3": "INTEGER",
		  "rel_w_usage": "INTEGER",
		  "longest_rtime": "INTEGER",
		  "sum_rtimes": "BIGINT",
		  "break_time": "INTEGER",
		  "submit_tday_sin": "REAL",
		  "submit_tday_cos": "REAL",
		  "submit_tweek_sin": "REAL",
		  "submit_tweek_cos": "REAL",
		  "avg_norm_ncpus": "REAL",
		  "avg_norm_mem": "REAL",
		  "cavg_ncpus": "REAL",
		  "cavg_mem": "REAL",
		  "occ_ncpus_total": "INTEGER",
		  "occ_mem_total": "INTEGER",
		  "running_jobs": "INTEGER",
		  "last_killed_num": "INTEGER",
		  "last_killed": "BOOL"}
    _label = {"run_time": "INTEGER"}

    @abc.abstractmethod
    def _queuejob_data(self, submit_time):
        """ Returns list of constructed predictors without scaling.
            The order from _predictors above must be preserved!
            Parameters
            ----------
            submit_time - time when the the job was submitted
        """
        return 

    @abc.abstractmethod
    def _modifyjob_data(self, data):
        """ Returns list of modified predictors without scaling
            The order from _predictors above must be preserved!
            Parameters
            ----------
            data - dict of job predictors fetched from database
        """
        return

class EventHandler(DPInterface):
    _meta = {"jid": "SERIAL",
            "pbs_jid": "VARCHAR",
            "exit_status": "SMALLINT",
            "start_time": "INTEGER",
            "submit_time": "INTEGER",
            "sw_pred": "INTEGER"}

    _fallback_mapper = {"avg_2": AvgTwo,
                        "rel_w_usage": RelativeWalltimeUsage}
    _jcheckpoints = sql.Identifier("jcheckpoints")

    def __init__(self,
                 storage,
                 repository,
                 threshold,
                 fallback,
                 jfit_limit=False,
                 jperiod=None,
                 jrate=None):
        self.storage = storage
        self.threshold = threshold
        self.repository = repository
        self.fallback = fallback
        self.jfit_limit = jfit_limit
        self.jperiod = jperiod
        self.jrate = jrate

    @property
    def label(self):
        return list(self._label.keys())

    @property
    def meta(self):
        return list(self._meta.keys())

    @property
    def predictors(self):
        return list(self._predictors.keys())

    @property
    def attributes(self):
        return self.predictors + self.meta + self.label

    @property
    def minput(self):
        return self.predictors + self.label

    @property
    def catmask(self):
        catmask = np.repeat(False, len(self.predictors))
        catmask[-1:] = True
        return catmask

    def handle(self):
        try:
            self.event = pbs.event()
            self.job = self.event.job
            if "jid" not in self.job.Resource_List or \
               "jowner" not in self.job.Resource_List:
                _accept(self.event, \
                        "Missing resource definition for 'jid' and 'jowner'!", \
                        loglevel=pbs.LOG_WARNING)
            if self.event.job is None:
                _reject(self.event, "Unset 'job' inside event!")
        
            self.conn = connect(**self.storage)
            self.cursor = self.conn.cursor()
            if self.event.type == pbs.QUEUEJOB:
                self.handle_queuejob()

		    # If the hook was inserted into system when the jobs were already
            # running, it may happen, that we encounter job without jid. Do not
            # collect any information about such jobs
            if self.job.Resource_List["jid"] is None and \
               self.event.type != pbs.MODIFYJOB:
                _accept(self.event, "Job was submitted before the hook import, skipping")

            if self.event.type == pbs.MODIFYJOB:
                self.job_o = self.event.job_o
                if self.job_o is None:
                    _reject(self.event, "Unset 'job_o' in modifyhook event!")
                if self.job_o.Resource_List["jid"] is None:
                    _accept(self.event, "Modify job submitted before the hook import, skipping")
                self.handle_modifyjob()
            if self.event.type == pbs.RUNJOB:
                self.handle_runjob()
            if self.event.type == pbs.EXECJOB_END:
                self.handle_execjob_end()
                
            _accept(self.event, msg="can be associated only with the QUEUEJOB, "\
                    + "MODIFYJOB, RUNJOB and EXECJOB_END events!",\
                    loglevel=pbs.LOG_WARNING)

        except SystemExit:
            pass
	
    def handle_queuejob(self):
        self.cursor.execute(sql.SQL(
            "CREATE TABLE IF NOT EXISTS {} ("
            + ", ".join(["%s\t%s" % (a, b) for a, b in self._predictors.items()]) 
            + ", " + ", ".join(["%s\t%s" % (a, b) for a, b in self._meta.items()])
            + ", " + ", ".join(["%s\t%s" % (a, b) for a, b in self._label.items()])
            + ")"
		).format(sql.Identifier(self.event.requestor)))
        self.conn.commit()
        submit_time = time.time()
        self.job.Resource_List["jowner"] = self.event.requestor

        model = self.prepare_model(submit_time)
        predictors = self._queuejob_data(submit_time)
        if model.is_fallback():
            y_pred = model.predict(self.job.Resource_List["walltime"])
        else:
            X = np.array(predictors, ndmin=2)
            X = self._poly(X)
            y_pred = model.predict(X)[0]
        self.job.Resource_List["soft_walltime"] = self._get_sw(y_pred)

        self.cursor.execute(sql.SQL(
            "INSERT INTO {}"
            + " (" + ", ".join(self.predictors + ["submit_time", "sw_pred"]) + ")"
            + " VALUES (" + ", ".join(["%s"] * (len(self.predictors) + 2)) + ")"
            + " RETURNING jid"
        ).format(sql.Identifier(self.job.Resource_List["jowner"])),
            (*predictors, submit_time, math.ceil(y_pred)))
        self.job.Resource_List["jid"] = self.cursor.fetchall()[0][0]
        self.conn.commit()
        _accept(self.event, "Queuejob handle has finished")

    def _queuejob_data(self, submit_time):
        # This dict construction forces order
        data = {key : None for key in self.predictors}
        data.update(self._cyclic(submit_time))
        data.update(self._killed(submit_time))
        data.update(self._running(submit_time))
        data.update(self._runtime(submit_time))
        data.update(self._sw_predictor(submit_time))
        data.update(self._hw_resources())
        
        walltime = self.job.Resource_List["walltime"]
        data["walltime"] = int(walltime) if walltime else walltime
        ncpus = self.job.Resource_List["ncpus"]
        data["ncpus"] = int(ncpus) if ncpus else ncpus
        mem = self.job.Resource_List["mem"]
        data["mem"] = size_as_int(mem, "kb") if mem else mem
        return list(data.values())

    def handle_modifyjob(self):
        if self.event.requestor in ["PBS_Server", "Scheduler", "pbs_mom"]:
            _accept(self.event, f"Modifyjob accepted for event requestor {self.event.requestor}")
        
        if self.job_o.stime and self.job_o.stime < time.time():
            _accept(self.event, "Modifyjob accepted, no change in soft walltime"\
                 + " because job is already running")

        # Non-priviliged user cannot change or see(flag=i) these by default. However,
        # the root user is able to make changes. The following lines prevent these changes
        self.job.Resource_List["jid"] = self.job_o.Resource_List["jid"]
        self.job.Resource_List["jowner"] = self.job_o.Resource_List["jowner"]
        
        # The eviscerated event.job object contains only changed resources. Since 
        # there is no documented way to create job object with the combination of
        # job_o and changed job attrs, populate job with the resources from job_o
        change = False
        for resource in ["walltime", "ncpus", "mem"]:
            res_o = self.job_o.Resource_List[resource]
            if self.job.Resource_List[resource] is None or \
               not self.job.Resource_List[resource]:
                self.job.Resource_List[resource] = res_o
            elif self.job.Resource_List[resource] != res_o:
                change = True

        if not change:
            _accept(self.event, f"Modifyjob accepted, no change in resources")
            
        self.cursor.execute(sql.SQL(
            "SELECT " + ", ".join(self.predictors + ["submit_time"])
            + " FROM {}"
            + " WHERE jid = %s"
        ).format(sql.Identifier(self.job.Resource_List["jowner"])),
            (self.job.Resource_List["jid"],))
        data = self.cursor.fetchall()[0]
        submit_time = data[-1]
        data = dict(zip(self.predictors, data[:-1]))
        predictors = self._modifyjob_data(data, submit_time)
        
        model = self.prepare_model(submit_time)
        if model.is_fallback():
            y_pred = model.predict(self.job.Resource_List["walltime"])
        else:
            X = np.array(predictors, ndmin=2)
            X = self._poly(X)
            y_pred = model.predict(X)[0]

        self.cursor.execute(sql.SQL(
            "UPDATE {} SET "
            + ", ".join(["{} = %s" for i in range(len(self.predictors) + 1)])
            + " WHERE jid = %s"
        ).format(sql.Identifier(self.job.Resource_List["jowner"]), \
        *[sql.Identifier(pred) for pred in self.predictors], sql.Identifier("sw_pred")), \
			(*predictors, math.ceil(y_pred), self.job.Resource_List["jid"]))
        self.conn.commit()

        self.job.Resource_List["soft_walltime"] = self._get_sw(y_pred)
        mtype = 'fallback' if model.is_fallback() else 'model'
        _accept(self.event, f"Modifyjob handle has finished ({mtype})")

    def _modifyjob_data(self, data, submit_time):
        walltime = self.job.Resource_List["walltime"]
        walltime_o = self.job_o.Resource_List["walltime"]
        if walltime != walltime_o:
            data.update(self._sw_predictor(submit_time))

        ncpus = self.job.Resource_List["ncpus"]
        ncpus_o = self.job_o.Resource_List["ncpus"]

        mem = self.job.Resource_List["mem"]
        mem = size_as_int(mem, "kb") if mem else mem

        mem_o = self.job_o.Resource_List["mem"]
        mem_o = size_as_int(mem_o, "kb") if mem_o else mem_o

        if mem != mem_o or ncpus != ncpus_o:
            data.update(self._hw_resources())

        data["mem"] = mem
        data["ncpus"] = ncpus
        data["walltime"] = walltime
        return list(data.values())
    
    def handle_runjob(self):
        self.cursor.execute(sql.SQL(
            "UPDATE {} SET start_time = %s, pbs_jid = %s WHERE jid = %s"
        ).format(sql.Identifier(self.job.Resource_List["jowner"])), \
            (time.time(), self.job.id, self.job.Resource_List["jid"]))
        self.conn.commit()
        _accept(self.event, "Runjob handle has finished")

    def handle_execjob_end(self):
        if not self.job.in_ms_mom():
             _accept(self.event, "Execjob_end on non-ms Mom, accepting")
        self.cursor.execute(sql.SQL(
            """UPDATE {}
               SET run_time = %s, exit_status = %s, pbs_jid = %s
               WHERE jid = %s"""
        ).format(sql.Identifier(self.job.Resource_List["jowner"])),\
            (self.job.resources_used["walltime"], self.job.Exit_status,\
             self.job.id, self.job.Resource_List["jid"]))
        self.conn.commit()
        pbs.logmsg(pbs.LOG_DEBUG, (f"Hook {self.event.hook_name}: Recording"
            f" run_time {self.job.resources_used['walltime']} and exit_status"
            f" {self.job.Exit_status} for job {self.job.id}"))
        _accept(self.event, "Execjob_end handle has finished")

    def _construct_ml_model(self, till):
        import pickle
        try:
            mpath = os.path.join(self.repository, self.job.Resource_List["jowner"])
            since = os.path.getmtime(mpath)
            with open(mpath, 'rb') as mfile:
                model = pickle.load(mfile)
            # This can happen during modifyjob event
            if since >= till:
                return model
        except FileNotFoundError:
            since = 0
            model = PolyRegressor(lrate=0.4, alpha=0.1)

        # Apply rate limit from 
        if self.jfit_limit and not self._allow_fit():
            pbs.logmsg(pbs.LOG_DEBUG, "Fit rate limit was exceeded, returning model!")
            return model

        self.cursor.execute(sql.SQL(
            "SELECT " + ", ".join(self.minput)
            + " FROM {}"
            + """ WHERE exit_status = 0 AND mem IS NOT NULL
                  AND ncpus IS NOT NULL AND walltime IS NOT NULL"""
            + (f" AND {since} <= (start_time + run_time)" if since else "")
            + " AND %s > (start_time + run_time)"
            + " ORDER BY %s - (start_time + run_time) DESC"
        ).format(sql.Identifier(self.job.Resource_List["jowner"])), (till, till))
        try:
            # If there are no new data for model training, make sure
            # that the loaded model was trained at least once
            if self.cursor.rowcount < 1:
                check_is_fitted(model)
                return model
        except NotFittedError:
            _reject(self.event, "Cannot use non-fitted model!")

        data = np.array(self.cursor.fetchall())
        X, y = np.hsplit(data, [len(self.predictors)])
        X = self._poly(X)
        y = y.reshape((y.size,))
        model.fit(X, y)
        with open(mpath, 'wb') as mfile:
            pickle.dump(model, mfile)
        return model

    def prepare_model(self, time_point):
        # Entries that can be used to fit the fallback or ML model
        req_hwalltime = self._fallback_mapper[self.fallback].req_hwalltime
        self.cursor.execute(sql.SQL(
            "SELECT COUNT(" + ("walltime IS NOT NULL" if req_hwalltime else "*") + "), "
               + """COUNT(walltime IS NOT NULL AND
                          mem IS NOT NULL AND
                          ncpus IS NOT NULL)
               FROM {}
               WHERE exit_status = 0 AND %s > (start_time + run_time)"""
        ).format(sql.Identifier(self.job.Resource_List["jowner"])), (time_point,))
        fcount, mcount = self.cursor.fetchall()[0]
        pbs.logmsg(pbs.LOG_DEBUG, f"Available entries for Model: {mcount}, Fallback: {fcount}")
        force_fallback = self.job.Resource_List["walltime"] is None or\
                         self.job.Resource_List["ncpus"] is None or\
                         self.job.Resource_List["mem"] is None
        if mcount >= self.threshold and not force_fallback:
            model = self._construct_ml_model(time_point)
            pbs.logmsg(pbs.LOG_DEBUG, "Deploying regression model!")
            return model
        try:
            limit = min(fcount, self._fallback_mapper[self.fallback].entries)
            self.cursor.execute(sql.SQL(
                "SELECT walltime, run_time"
                + " FROM {}"
                + " WHERE exit_status = 0" + (" AND walltime IS NOT NULL" if req_hwalltime else "")
                + " ORDER BY %s - (start_time + run_time) ASC"
                + " LIMIT %s"
            ).format(sql.Identifier(self.job.Resource_List["jowner"])),\
               (time_point, limit))
            walltime, run_time = zip(*self.cursor.fetchall())
        except:
            if self.job.Resource_List["walltime"] is None:
                _accept(self.event, "There are no finished jobs without missing walltimes"\
                          + " and the current job walltime estimate is missing, skipping")
            pbs.logmsg(pbs.LOG_DEBUG, "Deploying requested walltime!")
            return Walltime()
        if self._fallback_mapper[self.fallback].req_cwalltime \
               and self.job.Resource_List["walltime"] is None:
            model = AvgTwo()
            pbs.logmsg(pbs.LOG_DEBUG, "Deploying average of the last 2 job runtimes!")
        else:
            model = self._fallback_mapper[self.fallback]()
            pbs.logmsg(pbs.LOG_DEBUG, f"Deploying configured fallback: {self.fallback}")
        pbs.logmsg(pbs.LOG_DEBUG, f"Fitting model with walltime: {walltime}, run_time: {run_time}")
        model.fit(np.array(walltime), np.array(run_time))
        return model

    def _poly(self, X, y=None):
        pf = PolynomialFeatures()
        X_poly = pf.fit_transform(X[:, ~self.catmask])
        X = np.concatenate([X_poly, X[:, self.catmask]], axis=1)
        return X

    def _get_sw(self, y_pred):
        cwalltime = self.job.Resource_List["walltime"]
        y_pred = math.ceil(y_pred)
        if y_pred < 0:
            y_pred = cwalltime
            
        sw = pbs.duration(y_pred)
        if cwalltime and sw > cwalltime:
            sw = cwalltime
            pbs.logmsg(pbs.LOG_DEBUG, (f"Hook {self.event.hook_name}: Setting soft"
                f" walltime to hard walltime because it is larger ({sw} > {cwalltime})"))
        return sw 

    def _allow_fit(self):
        # Create table for checkpoints if it does not exists
        self.cursor.execute(sql.SQL(
            """CREATE TABLE IF NOT EXISTS {} (
               jowner varchar(256),
               jcheckpoint integer)
            """
        ).format(self._jcheckpoints))
        self.conn.commit()

        now = time.time()
        jowner = self.job.Resource_List["jowner"]
        self.cursor.execute(sql.SQL(
            "SELECT jcheckpoint FROM {} WHERE jowner = %s"
        ).format(self._jcheckpoints), (jowner,))
        if self.cursor.rowcount < 1:
            jcheckpoint = now
            self.cursor.execute(sql.SQL(
                "INSERT INTO {}(jowner, jcheckpoint) VALUES(%s, %s)"
            ).format(self._jcheckpoints), (jowner, jcheckpoint))
            self.conn.commit()
            
        if now - self.jperiod > jcheckpoint:
            offset = int((now - jcheckpoint) / self.jperiod) * self.jperiod
            jcheckpoint += offset
            pbs.logmsg(pbs.LOG_DEBUG, f"Updating {jowner}'s job checkpoint to {jcheckpoint}")
            self.cursor.execute(sql.SQL(
                "UPDATE {} SET jcheckpoint = %s WHERE jowner = %s"
            ).format(self._jcheckpoints), (jcheckpoint, jowner))
            self.conn.commit()

        self.cursor.execute(sql.SQL(
            """ SELECT COUNT(*)
                FROM {}
                WHERE submit_time > %s
            """
        ).format(sql.Identifier(jowner)), (jcheckpoint,))
        jobs = self.cursor.fetchall()[0]
        if self.jrate <= jobs:
            pbs.logmsg(pbs.LOG_DEBUG, f"Skipping training, limit exceeded {self.jrate} <= {jobs}")
            return False
        return True
    
    ############### Helper functions to implement DPInterface #################
    
    
    def _cyclic(self, submit_time):
        tday = 86400
        tweek = 604800
        day_angle = (2 * math.pi / tday) * (submit_time % tday)
        week_angle = (2 * math.pi / tweek) * (submit_time % tweek)
        return {"submit_tday_sin": math.sin(day_angle),
                "submit_tday_cos": math.cos(day_angle),
                "submit_tweek_sin": math.sin(week_angle),
                "submit_tweek_cos": math.cos(week_angle)}

    def _killed(self, submit_time):
        self.cursor.execute(sql.SQL(
            """SELECT exit_status, last_killed_num
               FROM {}
               WHERE exit_status = 0 OR run_time >= walltime
               ORDER BY %s - (start_time + run_time) ASC
               LIMIT %s"""
        ).format(sql.Identifier(self.job.Resource_List["jowner"])), (submit_time, 1))
        if self.cursor.rowcount < 1:
            last_killed, last_killed_num = False, 0
        else:
            exit_status, last_killed_num = self.cursor.fetchall()[0]
            last_killed = exit_status != 0
            last_killed_num = last_killed_num + int(exit_status != 0)

        return {"last_killed": last_killed,
                "last_killed_num": last_killed_num}

    def _running(self, submit_time):
        self.cursor.execute(sql.SQL(
            """SELECT start_time, ncpus, mem, pbs_jid
               FROM {}
               WHERE run_time IS NULL AND start_time < %s"""
        ).format(sql.Identifier(self.job.Resource_List["jowner"])), (submit_time,))
        sum_rtimes, longest_rtime = 0, 0
        occ_ncpus_total, occ_mem_total = 0, 0
        running_jobs = 0
        for stime, ncpus, mem, pbs_jid in self.cursor:
            job = pbs.server().job(pbs_jid)
            # Handle faulty running jobs in DB, that are
            # unknown to system by their skipping
            if job is None:
                continue
            # Handle case when the job has already finished, but the execjob_end
            # has not been called because transition to JOB_STATE_FINISHED takes time
            if job.job_state == pbs.JOB_STATE_EXITING and \
               stime + job.resources_used["walltime"] < submit_time:
                continue
            sum_rtimes += (submit_time - stime)
            longest_rtime = max(submit_time - stime, longest_rtime)
            occ_ncpus_total += ncpus if ncpus else 0
            occ_mem_total += mem if mem else 0
            running_jobs += 1
	
        return {"sum_rtimes": sum_rtimes,
                "longest_rtime": longest_rtime,
                "occ_ncpus_total": occ_ncpus_total,
                "occ_mem_total": occ_mem_total,
                "cavg_ncpus": occ_ncpus_total / max(running_jobs, 1),
                "cavg_mem": occ_mem_total / max(running_jobs, 1),
                "running_jobs": running_jobs}

    def _sw_predictor(self, submit_time):
        self.cursor.execute(sql.SQL(
            """ SELECT walltime, run_time
                FROM {}
                WHERE exit_status = 0 AND
                      %s > (start_time + run_time) AND
                      walltime IS NOT NULL
                ORDER BY %s - (start_time + run_time) ASC
                LIMIT %s
           """
        ).format(sql.Identifier(self.job.Resource_List["jowner"])), \
            (submit_time, submit_time, 5))
        cwalltime = self.job.Resource_List["walltime"]
        if self.cursor.rowcount < 1 or cwalltime is None:
            return {"rel_w_usage": cwalltime}
        data = np.array(self.cursor.fetchall())
        walltime, runtime = np.hsplit(data, [1])
        wall_usage = np.max(runtime / walltime)
        return {"rel_w_usage": math.ceil(wall_usage * cwalltime)}

    def _runtime(self, submit_time):
        predictors = ["last_1", "last_2", "last_3", "avg_2", "avg_3", "break_time"]
        self.cursor.execute(sql.SQL(
            """ SELECT start_time + run_time, run_time
                FROM {}
                WHERE exit_status = 0 AND
                      %s > (start_time + run_time)
                ORDER BY %s - (start_time + run_time) ASC
                LIMIT %s"""
        ).format(sql.Identifier(self.job.Resource_List["jowner"])), \
            (submit_time, submit_time, 5))
        jobs = self.cursor.rowcount
        if jobs < 1:
            cwalltime = self.job.Resource_List["walltime"]
            res = dict(zip(predictors, [cwalltime] * len(predictors)))
            res["break_time"] = 0
            return res
        end_time, run_time = zip(*self.cursor.fetchall())
        return {"last_1": run_time[0],
                "last_2": run_time[1] if jobs > 1 else run_time[-1],
                "last_3": run_time[2] if jobs > 2 else run_time[-1],
                "avg_2": round(sum(run_time[:2]) / len(run_time[:2])),
                "avg_3": round(sum(run_time[:3]) / len(run_time[:3])),
                "break_time": submit_time - end_time[0]}

    def _hw_resources(self):
        self.cursor.execute(sql.SQL(
            "SELECT AVG(ncpus), AVG(mem) FROM {}"
        ).format(sql.Identifier(self.job.Resource_List["jowner"])))
        avg_ncpus, avg_mem = self.cursor.fetchall()[0]
        ncpus = self.job.Resource_List["ncpus"]
        mem = self.job.Resource_List["mem"]
        mem = size_as_int(mem, "kb") if mem else mem

        avg_ncpus = avg_ncpus if avg_ncpus else ncpus
        avg_norm_ncpus = float(ncpus / avg_ncpus) if ncpus else None

        avg_mem = avg_mem if avg_mem else mem
        avg_norm_mem = float(mem / avg_mem) if mem else None
        return {"avg_norm_mem": avg_norm_mem, "avg_norm_ncpus": avg_norm_ncpus}

    def print_attribs(self, pbs_obj):
        for a in pbs_obj.attributes:
            v = getattr(pbs_obj, a)
            if v and str(v) != "":
                pbs.logmsg(pbs.LOG_DEBUG, "%s = %s" % (a,v))


DPInterface.register(EventHandler)


def check_config(event, config):
    """ Check whether loaded configuration file contains
        all the necessary information for DB, model training"""

    config.setdefault("storage", {})
    if not isinstance(config["storage"], dict):
        _reject(event, "DB 'storage' configuration is not dictionary!")
    if "dbname" not in config["storage"]:
        _reject(event, "DB storage configuration is missing 'dbname'!")
    if "host" not in config["storage"]:
        _reject(event, "DB storage configuration is missing 'host'!")

	# Check presence of model repository
    if "repository" not in config:
        config["repository"] = os.path.join(pbs.pbs_conf["PBS_HOME"], "sched_priv/models")
    if not os.path.exists(config["repository"]):
        os.mkdir(config["repository"])
    if not os.path.isdir(config["repository"]):
        _reject(event, "Provided model 'repository' is not directory")

	# Check threshold and its value
    config.setdefault("threshold", 20)
    if not isinstance(config["threshold"], int):
        _reject(event, "Threshold configuration option should be int!")
    elif config["threshold"] < 1:
        _reject(event, "Threshold configuration option should be > 0!")

	# Check fallback technique setting
    config.setdefault("fallback", "avg_2")
    if config["fallback"] not in ["avg_2", "rel_walltime_usage"]:
        _reject(event, "Invalid 'fallback' technique configuration!")

    # Check rate limit for model fitting
    config.setdefault("jfit_limit", False)
    if not isinstance(config["jfit_limit"], bool):
        _reject(event, "Invalid type for bool 'jfit_limit' option!")
    if config["jfit_limit"]:
        if "jperiod" not in config or "jrate" not in config:
            _reject(event, "Missing 'jrate' or 'jperiod' for enabled 'jfit_limit'!")
        if not isinstance(config["jperiod"], int):
            _reject(event, "Option 'jperiod' should be integer!")
        if not isinstance(config["jrate"], int):
            _reject(event, "Option 'jrate' should be integer!")
    return config

def read_config(event):
    """ Read the hook configuration file in json format"""
    import simplejson
    try:
        pbs_hook_cfg = pbs.hook_config_filename
        if pbs_hook_cfg is None:
            pbs_hook_cfg = os.environ["PBS_HOOK_CONFIG_FILE"]
        with open(pbs_hook_cfg, "r") as config_file:
            config = simplejson.load(config_file)

    except simplejson.JSONDecodeError:
        _reject(event, f"Could not parse hook configuration {pbs_hook_cfg}", trace_exc=True)

    return config

try:
    # This allows the pickle module to store/load model
    builtins.PolyRegressor = PolyRegressor
    event = pbs.event()
    config = read_config(event)
    config = check_config(event, config)
    eh = EventHandler(**config)
    eh.handle()

except:
    _reject(pbs.event(), trace_exc=True)
