# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 12:44:36 2023

@author: Grady
"""

import pandas as pd
from statsmodels.sandbox.regression.gmm import GMM, GMMResults
from statsmodels.regression.linear_model import WLS
import numpy as np 
from patsy import dmatrix, ModelDesc, DesignInfo
from math import cos, pi, log, pi
from sklearn.linear_model import LassoCV 
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold 


class spatialGMM(GMM):
    """
    Custom GMM class that incorporates Conley SEs. Extends the statsmodels GMM class, and supports most functionality 
    from the package. 
    ----------
    latitude: A numpy array containing the latitude points for spatial SE calculations.
    longitude: A numpy array containing the longitude points for spatial SE calculations.
    kernel (str default 'uniform'): If 'uniform' uses a uniform kernel. If 'bartlett' uses a Bartlett kernel. If 'pos_def' uses (1 - dist/dist_cutoff)^2 to ensure positive definite covariance. 
    distance_cutoff: (Optional, default 10) Distance cutoff for spatial SEs in km.
    """
    
    def __init__(self, endog, exog, instrument, latitude, longitude, k_moms=None, k_params=None,
                 missing='none', kernel='uniform', distance_cutoff=10, **kwds):
        
        instrument = self._check_inputs(instrument, endog, latitude, longitude) # attaches if needed
        self.kernel = self._calc_kernel(latitude, longitude, distance_cutoff, kernel=kernel)  # Done once here so not recalculated on each iteration
        super(GMM, self).__init__(endog, exog, missing=missing,
                instrument=instrument)
#         self.endog = endog
#         self.exog = exog
#         self.instrument = instrument
        self.nobs = endog.shape[0]
        if k_moms is not None:
            self.nmoms = k_moms
        elif instrument is not None:
            self.nmoms = instrument.shape[1]
        else:
            self.nmoms = np.nan

        if k_params is not None:
            self.k_params = k_params
        elif instrument is not None:
            self.k_params = exog.shape[1]
        else:
            self.k_params = np.nan

        self.__dict__.update(kwds)
        self.epsilon_iter = 1e-6

    def _check_inputs(self, instrument, endog, latitude, longitude):
        if latitude.shape[0] != endog.shape[0]:
            raise ValueError("Latitude not the same length as endog")
        if longitude.shape[0] != endog.shape[0]:
            raise ValueError("Longitude not the same length as endog")
        if instrument is not None:
            offset = np.asarray(instrument)
            if offset.shape[0] != endog.shape[0]:
                raise ValueError("instrument is not the same length as endog")
        return instrument
    
    def _calc_kernel(self, lat, lon, cutoff=10, kernel='uniform'):
        K = np.empty([len(lat), len(lat)])
        for i in range(len(lat)):
            lati = lat.iloc[i]
            loni = lon.iloc[i]
            distancei = self._calc_distance(lati, loni, lat, lon)
            if kernel == 'uniform':
                Ki = [1 if d <= cutoff else 0 for d in distancei]
            elif kernel == 'bartlett':
                Ki = [1 - d/cutoff if d <= cutoff else 0 for d in distancei]
            elif kernel == 'pos_def':
                Ki = [(1 - d/cutoff)**2 if d <= cutoff else 0 for d in distancei]
            K[i,:] = Ki 
        return K 
    
    # Define a function for calculating the distance between coordinates in km
    ## Adapted from Sol Hsiang's code for consistency 
    def _calc_distance(self, lati, loni, lat, lon):
        lon_scale = cos(lati*pi/180)*111
        lat_scale = 111
        distancei = ((lat_scale*(lati - lat))**2 + (lon_scale*(loni - lon))**2)**0.5
        return distancei
    
    
    def calc_weightmatrix(self, moms, df_correction=None, weights_method='conley', wargs=None, params=None):
        # weights_method, params and wargs to avoid changing other statsmodels code, not used but don't delete
        nobs, k_moms = moms.shape
        
        V = np.zeros([k_moms, k_moms])

        for i in range(nobs):
            K_i = np.array([self.kernel[i,:]]).T
            g_i = np.tile(np.array([moms[i, :]]).T, nobs)
            g_j = moms * K_i  # Elementwise multiplication
            V_i = (1/nobs)*np.dot(g_i, g_j) 
            V = V + V_i
            
        return V


class geGMM(spatialGMM):
    """
    Custom GMM class that incorporates Conley SEs and is customized for ease of use on the GE project.
    Parameters
    ----------
    data: Pandas dataframe 
    dep_vars: List of dependent variables.  
    exog: A Patsy formula specifying which exogenous variables (x) to include 
    instruments: A Pasty formula specifying which instruments (z) to include 
    moments: (Optional) List of moment conditions. Each moment condition should be a string where parameters are 
        accessed using the list p, instruments using the list z, and exogenous variables using x. If not set 
        assumes each exogenous variable, instrument should be used linearly for every endogenous variable.
    latitude: Variables name containing latitude 
    longitude: Variable name containing longitude 
    pweights: (Optional) variable name containing inverse probability weights 
    other_vars: (Optional) list of other variables to append to data, without partialling out controls (e.g. sample indicators) 
    controls: (Optional) Patsy formula with controls to partial out 
    distance_cutoff: Default 10km: spatial distance cutoff for Conley error estimation in km
    kernel (str default 'uniform'): If 'uniform' uses a uniform kernel. If 'bartlett' uses a Bartlett kernel. If 'pos_def' uses (1 - dist/dist_cutoff)^2 to ensure positive definite covariance. 
    parameters: A list of parameter names, as strings. Optional if simple IV case. Required if using other moments.
    lasso_controls: A Patsy formula specifying any controls to select from using double partial out LASSO
    lasso_seed: Random seed for 5-fold CV when specifying LASSO controls. Default 1.
    lasso_exclude: Base levels excluded from LASSO inclusion. Default None

    Attributes
    ----------
    n_exog : int
        Number of exogenous variables
    n_endog : int
        Number of endogenous variables
    n_inst : int 
        Number of instruments 
    weights : ndarray
        Inverse probability weights, if specified 
    moments : list
        List of moment conditions
    instrument_names: list 
        List of instrument names
    exog_names: list 
        List of exogenous variable names
    parameters: list 
        List of parameter names, if included
    other_vars: ndarray
        Other variables passed to class

    Methods
    -------
    _partialOutControls(data, var_list, controls, weight_name=None)
        Partials out controls from each variable in var_list. Used internally.
    """
    
    def __init__(self, data, dep_vars, exog, instruments, moments=None, 
                 latitude='latitude', longitude='longitude', pweights=None, other_vars=None,
                 controls=None, distance_cutoff=10, kernel='uniform', parameters=None, lasso_controls=None, lasso_seed=1, lasso_exclude=None, *args, **kwds):

        # If dep_vars specified as a string, convert to a list 
        if isinstance(dep_vars, str):
            dep_vars = dep_vars.split(',')

        self.lasso_seed = lasso_seed
        self.lasso_exclude = lasso_exclude
        
        # Get an initial list of variables to drop 
        if (controls is not None) and (lasso_controls is not None):
            full_controls = controls + ' + ' + lasso_controls
        elif controls is not None:
            full_controls = controls
        elif lasso_controls is not None:
            full_controls = lasso_controls
        else: 
            full_controls = None
        if full_controls is not None:
            md = ModelDesc.from_formula('{} ~ {} + {} + {}'.format(dep_vars[0], exog, instruments, full_controls))
        else:
            md = ModelDesc.from_formula('{} ~ {} + {}'.format(dep_vars[0], exog, instruments))
        termlist = md.rhs_termlist 
        factors = []
        for term in termlist:
            for factor in term.factors:
                factors.append(factor.name())
                
        vars_to_use = list()
        vars_to_use = vars_to_use + dep_vars 
        for x in factors: 
            vars_to_use.append(x.replace('C(', '').replace(')', '').replace('np.log(', '').replace('center(', ''))  # Handles categorical variables, log transform and centering 
        if pweights is not None:
            vars_to_use = vars_to_use + [pweights]
        vars_to_use = vars_to_use + ['latitude', 'longitude']
        if other_vars is not None:
            vars_to_use = vars_to_use + other_vars
        vars_to_use = list(set(vars_to_use))  # Remove duplicates 
        
        _df = data.dropna(subset = vars_to_use)
        if other_vars is not None:
            self.other_vars = _df[other_vars].values
        else:
            self.other_vars = None
        
        if full_controls is not None:
            instruments = dmatrix(instruments + ' - 1', _df, return_type='dataframe')
            exog = dmatrix(exog + ' - 1', _df, return_type='dataframe')
        else:
            instruments = dmatrix(instruments, _df, return_type='dataframe')
            exog = dmatrix(exog, _df, return_type='dataframe')

        self.instrument_names = list(instruments.columns)    
        for x in instruments.columns:
            if x in exog.columns:
                instruments.drop(columns=[x], inplace=True)  # This avoids including controls twice in the final dataframe 
    
        if full_controls is not None:
            full_control_df = dmatrix(full_controls, _df, return_type='dataframe')
            df_full = pd.concat([_df[dep_vars], _df[[latitude, longitude]], instruments, exog, full_control_df], axis=1)

            for control in full_control_df.columns: 
                    if (control in exog.columns) or (control in instruments.columns) or (control.replace('[T.', '[') in exog.columns) or (control.replace('[T.', '[') in instruments.columns):  # the T. is for differences in naming levels with intercept
                        full_control_df.drop(columns=[control], inplace=True)  # Handles cases where accidentally list an included variable to partial out (e.g. when formula includes too many levels)

            if controls is not None:
                always_included_controls = dmatrix(controls, _df, return_type='dataframe').columns  # Captures always included controls, plus partials out first before running LASSO
                for control in always_included_controls: 
                    if (control in exog.columns) or (control in instruments.columns) or (control.replace('[T.', '[') in exog.columns) or (control.replace('[T.', '[') in instruments.columns):  # the T. is for differences in naming levels with intercept
                        always_included_controls.remove(control)  # Handles cases where accidentally list an included variable to partial out (e.g. when formula includes too many levels)
                if lasso_controls is not None:
                    potential_controls = list(dmatrix(lasso_controls, _df, return_type='dataframe').columns)
                    potential_controls = [x for x in potential_controls if x != 'Intercept']  # Ensures that still named relative to the intercept, excess levels aren't included 
                    for control in potential_controls: 
                        if (control in exog.columns) or (control in instruments.columns) or (control in always_included_controls) or (control.replace('[T.', '[') in exog.columns) or (control.replace('[T.', '[') in instruments.columns) or (control.replace('[T.', '[') in always_included_controls) or (control.replace('[', '[T.') in exog.columns) or (control.replace('[', '[T.') in instruments.columns) or (control.replace('[', '[T.') in always_included_controls):  # the T. is for differences in naming levels with intercept
                            potential_controls.remove(control)  # Handles cases where accidentally list an included variable to partial out (e.g. when formula includes too many levels)

            else:
                potential_controls = list(dmatrix(lasso_controls, _df, return_type='dataframe').columns)
                for control in potential_controls: 
                    if (control in exog.columns) or (control in instruments.columns) or (control.replace('[T.', '[') in exog.columns) or (control.replace('[T.', '[') in instruments.columns):  # the T. is for differences in naming levels with intercept
                        potential_controls = [x for x in potential_controls if x != control]  # Handles cases where accidentally list an included variable to partial out (e.g. when formula includes too many levels)
        else:
            df_full = pd.concat([_df[dep_vars], _df[[latitude, longitude]], instruments, exog], axis=1)

        if pweights is not None:
            df_full = pd.concat([df_full, _df[pweights]], axis = 1)
            df_full.dropna(inplace=True)
            
            if full_controls is not None:
                if controls is not None:
                    var_list = dep_vars + list(exog.columns) + list(instruments.columns)
                    lasso_var_list = dep_vars + list(exog.columns)  # Do not want to search for controls over instruments 
                    if lasso_controls is None:
                        self.n_controls = len(always_included_controls)
                        df_po = self._partialOutControls(data=df_full, var_list=var_list, controls=df_full[always_included_controls].values, weight_name=pweights)

                    else:
                        for x in combined_controls:
                            print(x)

                        df_po_temp = self._partialOutControls(data=df_full, var_list=var_list + list(potential_controls), controls=df_full[always_included_controls].values, weight_name=pweights)
                        lasso_selected = self._lassoSelect(included_vars=lasso_var_list, potential_controls=list(potential_controls), df=df_po_temp, weight_name=pweights)
                        del df_po_temp
                        combined_controls = list(always_included_controls) + lasso_selected
                        self.n_controls = len(combined_controls)
                        df_po = self._partialOutControls(data=df_full, var_list=var_list, controls=df_full[combined_controls].values, weight_name=pweights)
                else:
                    lasso_selected = self._lassoSelect(included_vars=lasso_var_list, potential_controls=list(potential_controls), df=df_full, weight_name=pweights)
                    self.n_controls = len(lasso_selected)
                    df_po = self._partialOutControls(data=df_full, var_list=var_list, controls=df_full[lasso_selected].values, weight_name=pweights)
            else:
                df_po = df_full 
                self.n_controls = 0 
        else:
            df_full.dropna(inplace=True)
            if full_controls is not None:
                if controls is not None:
                    var_list = dep_vars + list(exog.columns) + list(instruments.columns)
                    lasso_var_list = dep_vars + list(exog.columns)  # Do not want to search for controls over instruments 
                    if lasso_controls is None:
                        self.n_controls = len(always_included_controls)
                        df_po = self._partialOutControls(data=df_full, var_list=var_list, controls=df_full[always_included_controls].values)

                    else:
                        df_po_temp = self._partialOutControls(data=df_full, var_list=var_list + list(potential_controls), controls=df_full[always_included_controls].values)
                        lasso_selected = self._lassoSelect(included_vars=lasso_var_list, potential_controls=list(potential_controls), df=df_po_temp)
                        del df_po_temp
                        combined_controls = list(always_included_controls) + lasso_selected
                        self.n_controls = len(combined_controls)
                        df_po = self._partialOutControls(data=df_full, var_list=var_list, controls=df_full[combined_controls].values)
                else:
                    lasso_selected = self._lassoSelect(included_vars=lasso_var_list, potential_controls=list(potential_controls), df=df_full)
                    self.n_controls = len(lasso_selected)
                    df_po = self._partialOutControls(data=df_full, var_list=var_list, controls=df_full[lasso_selected].values)
            else:
                df_po = df_full 
                self.n_controls = 0 
            
        instrument = df_po[self.instrument_names].values
        endog = df_po[dep_vars].values
        self.exogenous_names = list(exog.columns)
        exog = df_po[exog.columns].values 
        if pweights:
            self.weights = df_full[[pweights]].values
        else:
            self.weights = None
            
        self.n_exog = exog.shape[1]
        self.n_inst = instrument.shape[1]
        self.n_endog = endog.shape[1]
        if parameters is not None:
            self.parameters = parameters  # List of parameter names, as string
            
        if moments is not None:
            self.moments = moments  # List of moment conditions, as string
        else:  # Build out moment conditions, assuming each instrument, exogenous variable used for each endogenous variable
            moments = list()
            if self.n_endog == 1:
                resid = '(y '
                for i in range(self.n_exog):
                    resid = resid + ' - params[{}]*x[{}]'.format(i, i) 
                resid = resid + ')'
                for i in range(self.n_inst):
                    prefix = 'z[{}]*'.format(i) 
                    moment = prefix + resid
                    moments.append(moment)
            else:
                for j in range(self.n_endog):
                    resid = '(y[{}] '.format(j)
                    for i in range(self.n_exog):
                        resid = resid + ' - params[{}]*x[{}]'.format(i, i) 
                    resid = resid + ')'
                    for i in range(self.n_inst):
                        prefix = 'z[{}]*'.format(i) 
                        moment = prefix + resid
                        moments.append(moment)
            self.moments = moments 
                
        self.pweights = pweights    
                        
        kwds.setdefault('k_moms', len(moments))
        if parameters is not None:
            kwds.setdefault('k_params', len(parameters))
        else:
            kwds.setdefault('k_params', self.n_exog)
        super(geGMM, self).__init__(endog = endog, exog = exog, instrument=instrument,
                                    latitude=df_full[latitude], longitude=df_full[longitude], 
                                    kernel=kernel, distance_cutoff=distance_cutoff, *args, **kwds)

        if parameters is not None:
            self.set_param_names(parameters)
        else:
            self.set_param_names(self.exogenous_names)
        
    
    def momcond(self, params):

        x = list()
        for i in range(self.n_exog):
            x.append(self.exog[:,i])
            
        z = list()
        for i in range(self.n_inst):
            z.append(self.instrument[:,i])
        
        if self.n_endog > 1:
            y = list()
            for i in range(self.n_endog):
                y.append(self.endog[:,i])
        else:
            y = self.endog
            
        if self.weights is not None:
            if len(self.weights.shape) > 1:  
                weight = self.weights[:,0]
            else:
                weight = self.weights
        else:
            weight = 1 

        if self.other_vars is not None:
            other_vars = list() 
            if len(self.other_vars.shape) > 1:
                for i in range(self.other_vars.shape[1]):
                    other_vars.append(self.other_vars[:,i])
            else:
                other_vars.append(self.other_vars)
            
        g = np.empty([self.exog.shape[0], len(self.moments)])
        j = 0
        
        for m in self.moments:
            g[:,j] = eval(m)*weight
            j = j + 1
        
        return g
    
    
    def _partialOutControls(self, data, var_list, controls, weight_name=None):
        df_po = pd.DataFrame(columns=var_list)
        _df = data.copy()
        _df.dropna(subset=var_list, inplace=True)
        if weight_name:
            weights = _df[weight_name]
            df_po[weight_name] = _df[weight_name]
        else:
            weights = 1
            
        for var in var_list:
            # Get the dependent variable data 
            _dep = _df[var].values
    
            # Regress the dependent variable on the fixed effects variables.
            ind_reg = WLS(_dep, controls, weights=weights).fit()
            df_po[var] = ind_reg.resid
            
        return df_po

    def _lassoSelect(self, included_vars, potential_controls, df, weight_name=None):        
        selected_controls = list() 

        if weight_name is not None:
            W = np.diag(np.sqrt(df[weight_name].values))  # Get weights manually for inverse probability weighting, since not supported natively by scikit learn

        for x in included_vars:
            if weight_name is not None:
                _y = np.dot(W, df[[x]])
            else:
                _y = df[[x]].values
            _y = _y.ravel()
            if weight_name is not None:
                _X = pd.DataFrame(columns=potential_controls, data=np.dot(W, df[potential_controls]))
            else:
                _X = df[potential_controls]
            lasso_cv = LassoCV(cv=KFold(n_splits=5, shuffle=True, random_state=self.lasso_seed)) 
            lasso_cv.fit(_X, _y)
            selected = SelectFromModel(lasso_cv, prefit=True) 
            feature_idx = selected.get_support()
            feature_names = _X.columns[feature_idx].tolist()
            selected_controls = selected_controls + feature_names
            
        # Add in base levels for any interactions included
        for x in selected_controls:
            if ':' in x:
                base_levels = x.split(':')
                for level in base_levels:
                    if (level in included_vars) or (level in self.instrument_names):
                        pass  # E.g. if lasso includes control * treat, but treat is a variable of interest already included 
                    elif (self.lasso_exclude is not None) and (level in self.lasso_exclude):
                        pass
                    else:
                        selected_controls.append(level)

        self.selected_controls = list(set(selected_controls))
        return list(set(selected_controls))  # Removes any duplicates 
        

def calculate_optimal_radii(data, endog, dynamic_exog, dynamic_instruments, static_instruments=None, 
                            static_exog=None, static_controls=None, dynamic_controls=None, monotonic=True,
                            latitude='latitude', longitude='longitude', kernel='uniform', distance_cutoff=10, pweights=None, maxiter=1,
                            optim_method='bfgs', optim_args=None, addBLcontrols='enterprise', tsls=False, lasso_controls=None, lasso_seed=1):
    """
    Figures out the optimal radius for a scalar outcome and a set of instruments by minimizing BIC. Only supports a single outcome.

    Args: 
        data (Pandas DataFrame): Data for estimation 
        endog (str): Name of endogenous variable 
        dynamic_exog (list of Patsy formulas): Dynamic exogenous variables in form of var_{}to{}km, where {} and {} will be filled with r-2 r km in loop. 
            These will be instrumented for (see dynamic_controls for optional controls)
        dynamic_instruments: (list of Patsy formulas) Dynamic instruments in form of var_{}to{}km, where {} and {} will be filled with r-2 r km in loop
        static_instruments: (optional str): Patsy string with any static instruments to include 
        static_controls: (optional str): Patsy string with any static controls (e.g. FEs) to include 
        dynamic_controls: (optional list of Patsy formulas): Spatially varying controls specified as Patsys formulas of var_{}to{}km, where {} and {} will be filled with r-2 r km in loop
        monotonic: (Default True). If True, stops once BIC increases and returns the prior value. Faster but less robust. 
        latitude (str): Name of latitude variable in data 
        longitude (str): Name of longitude variable in data
        kernel (str default 'uniform'): If 'uniform' uses a uniform kernel. If 'bartlett' uses a Bartlett kernel. If 'pos_def' uses (1 - dist/dist_cutoff)^2 to ensure positive definite covariance. 
        distance_cutoff (float default 10): Distance cutoff for kernel in km 
        pweights (optional, str): Name of column containing inverse probability weights in data 
        maxiter: (int default 1): Maximum number of GMM iterations. Default is 1, but if using 2 step GMM performance is better with set to 2.
        optim_method: (str, default 'bfgs') : scipy optimization method. bfgs is default for speed, 'nm' is more robust but slower. 
        optim_args: (optional, dict) : dictionary of arguments for the optimization algorithm (e.g. increase max iterations)
        addBLcontrols: (str, default 'enterprise') : searches for and adds BL controls if available. Default is to add enterprise BL controls.
            Currently only enterprise supported.
        tsls: (boolean, default False) If True, uses two-stage least squares weighting, and sets max iterations to 1 
        lasso_controls: A Patsy formula specifying any controls to select from using double partial out LASSO
        lasso_seed: Random seed for 5-fold CV when specifying LASSO controls. Default 1.

    Returns:
        opt_r, selected_exogenous, selected_instruments, selected_controls 

        opt_r is an integer with the selected maximum radius 
        optimal_exogenous is a Patsy string of the selected exogenous variables 
        optimal_instruments is a Patsy string with the selected set of instruments 
        optimal_controls is a Patsy string with the selected set of controls 
    """
    
    bic_values = dict()
    controls_dict = dict()
    exog_dict = dict() 
    instruments_dict = dict()

    if addBLcontrols == 'enterprise':
        # Only store (1) PPP version of variables in the enterprise data, (2) no r_ since real and nominal are equivalent at BL 
        endog_bl = endog.replace('r_', '')
        if '_PPP' not in endog_bl:
            if '_wins' in endog_bl:
                tmp = endog_bl.replace('_wins', '')
                endog_bl = tmp + '_PPP_wins_vBL'
            else:
                endog_bl = endog_bl + '_PPP_vBL'
        else:
            endog_bl = endog_bl + '_vBL'
        Mendog_bl = 'M' + endog_bl
        if (endog_bl in data.columns) and (Mendog_bl in data.columns): 
            if static_controls is None:
                static_controls = 'C(ent_type):{} + C(ent_type):{}'.format(endog_bl, Mendog_bl)
            else:
                static_controls = static_controls + ' + C(ent_type):{} + C(ent_type):{}'.format(endog_bl, Mendog_bl)
    
    for r in range(2, 22, 2):
        r2 = r - 2 
        if r == 2:
            add = 0 
            for inst in dynamic_instruments:
                if inst.count('{}')/2 == 1: 
                    rads = [r2, r]
                elif inst.count('{}')/2 == 2:
                    rads = [r2, r, r2, r]
                elif inst.count('{}')/2 == 2:
                    rads = [r2, r, r2, r, r2, r]
                if add == 0:
                    _dynamic_instrument_patsy = inst.format(*rads)
                    add = 1 
                else:
                    _dynamic_instrument_patsy = _dynamic_instrument_patsy + ' + ' + inst.format(*rads)
            add = 0 
            for x in dynamic_exog:
                if add == 0:
                    _dynamic_exog_patsy = x.format(r2, r)
                    add = 1 
                else:
                    _dynamic_exog_patsy = _dynamic_exog_patsy + ' + ' + x.format(r2, r)
            add = 0 
            if dynamic_controls is not None:
                for c in dynamic_controls:
                    if add == 0:
                        _dynamic_controls_patsy = c.format(r2, r)
                        add = 1 
                    else:
                        _dynamic_controls_patsy = _dynamic_controls_patsy + ' + ' + c.format(r2, r)
            
        else:
            if r > max_radius: # Stop execution and return the best performing value to that point 
                opt_r = min(bic_values, key=bic_values.get)
                return opt_r, exog_dict[opt_r], instruments_dict[opt_r], controls_dict[opt_r]   
            for inst in dynamic_instruments:
                if inst.count('{}')/2 == 1: 
                    rads = [r2, r]
                elif inst.count('{}')/2 == 2:
                    rads = [r2, r, r2, r]
                elif inst.count('{}')/2 == 2:
                    rads = [r2, r, r2, r, r2, r]
                _dynamic_instrument_patsy = _dynamic_instrument_patsy + ' + ' + inst.format(*rads)
            for x in dynamic_exog:
                _dynamic_exog_patsy = _dynamic_exog_patsy + ' + ' + x.format(r2, r)
                
            if dynamic_controls is not None:
                for c in dynamic_controls:
                    _dynamic_controls_patsy = _dynamic_controls_patsy + ' + ' + c.format(r2, r)
                
        if static_exog is not None:
            _exog = _dynamic_exog_patsy + ' + ' + static_exog
        else:
            _exog = _dynamic_exog_patsy
            
        if static_instruments is not None:
            _instrument = _dynamic_instrument_patsy + ' + ' + static_instruments
        else:
            _instrument = _dynamic_instrument_patsy
            
        if dynamic_controls is not None:
            if static_controls is not None:
                _controls = static_controls + ' + ' + _dynamic_controls_patsy
            else:
                _controls = _dynamic_controls_patsy
        else:
            _controls = static_controls
        
        _model = geGMM(data, [endog], _exog, _instrument, 
                     latitude=latitude, longitude=longitude, pweights=pweights, 
                     controls=_controls, distance_cutoff=distance_cutoff, kernel=kernel, lasso_controls=lasso_controls, lasso_seed=lasso_seed)
        
        _beta0 = np.ones(_model.n_exog)

        if tsls:
            _inv_w = (1/_model.instrument.shape[0])*np.dot(_model.instrument.T, _model.instrument) 
            _fitted = _model.fit(_beta0, inv_weights=_inv_w, maxiter=1, optim_method=optim_method)

        else:
            _fitted = _model.fit(_beta0, maxiter = maxiter, optim_method=optim_method)
        
        _pred = np.dot(_model.exog, _fitted.params) 
        _resid = _model.endog - _pred 
        if _model.weights is not None:
            _ssr = np.dot(_resid**2, _model.weights)[0] / np.mean(_model.weights)
            _sigma2 = np.average(_resid**2, weights=_model.weights[:,0])
        else:
            _ssr = np.dot(_resid**2, np.ones(_resid.shape[0]))
            _sigma2 = np.mean(_resid**2)
            
        _ll = -0.5*_model.exog.shape[0]*log(2 * pi * _sigma2) - 1/(2*pi*_sigma2)*_ssr 
        _bic = (_model.k_params + _model.n_controls)*log(_model.exog.shape[0]) - 2*_ll
        
        bic_values[r] = _bic
        controls_dict[r] = _controls 
        exog_dict[r] = _exog
        instruments_dict[r] = _instrument

        if r == 2: 
            if _model.exog.shape[0] < 1000:
                opt_r = 2 # Very few observations for these outcomes, so impose a max radius of 2 
                return opt_r, exog_dict[opt_r], instruments_dict[opt_r], controls_dict[opt_r]   
            elif _model.exog.shape[0] < 3000:
                max_radius = 8  # Try to prevent crashing 
            elif _model.exog.shape[0] < 5000:
                max_radius = 10
            else:
                max_radius = 20
        
        if r > 2 and monotonic:  # For speed, default assumes that the BIC won't improve if it is worsening, returns prior value 
            if _bic > bic_values[r-2]:
                opt_r = r - 2
                return opt_r, exog_dict[opt_r], instruments_dict[opt_r], controls_dict[opt_r]    
            
    opt_r = min(bic_values, key=bic_values.get)
    return opt_r, exog_dict[opt_r], instruments_dict[opt_r], controls_dict[opt_r]       


ent_type_weights_el2 = {1: .13313188, 2: .08802072, 3: .95598004}
def average_total_effect(ge_gmm_results, specification='enterprise', ent_type_weights=ent_type_weights_el2, el2_ent_data=None):
    params = ge_gmm_results.model.data.xnames 
    ATE_string_total = '0'
    ATE_string_spillover = '0'
    if specification == 'enterprise':
        if el2_ent_data is None:
            raise Exception('el2_ent_data must be specified if calculating enterprise results')
        for p in params:
            if 'C(ent_type)[1.0]:' in p:
                exog = p[len('C(ent_type)[1.0]:'):] 
                _var = el2_ent_data.loc[(~el2_ent_data[exog].isna()) & (el2_ent_data['treat'] == 1) & (el2_ent_data['ent_type'] == 1), exog]
                _weights = el2_ent_data.loc[(~el2_ent_data[exog].isna()) & (el2_ent_data['treat'] == 1) & (el2_ent_data['ent_type'] == 1), 'ent_weight_el2'] 
                _amt = np.average(_var, weights=_weights) 
                
                ATE_string_total = ATE_string_total + ' + {}*{}*{}'.format(ent_type_weights[1], _amt, p)
                
                _var = el2_ent_data.loc[(~el2_ent_data[exog].isna()) & (el2_ent_data['treat'] == 0) & (el2_ent_data['ent_type'] == 1), exog]
                _weights = el2_ent_data.loc[(~el2_ent_data[exog].isna()) & (el2_ent_data['treat'] == 0) & (el2_ent_data['ent_type'] == 1), 'ent_weight_el2'] 
                _amt = np.average(_var, weights=_weights) 
                
                ATE_string_spillover = ATE_string_spillover + ' + {}*{}*{}'.format(ent_type_weights[1], _amt, p)
                
            if 'C(ent_type)[2.0]:' in p:
                exog = p[len('C(ent_type)[2.0]:'):] 
                _var = el2_ent_data.loc[(~el2_ent_data[exog].isna()) & (el2_ent_data['treat'] == 1) & (el2_ent_data['ent_type'] == 2), exog]
                _weights = el2_ent_data.loc[(~el2_ent_data[exog].isna()) & (el2_ent_data['treat'] == 1) & (el2_ent_data['ent_type'] == 2), 'ent_weight_el2'] 
                _amt = np.average(_var, weights=_weights) 
                
                ATE_string_total = ATE_string_total + ' + {}*{}*{}'.format(ent_type_weights[2], _amt, p)
                
                _var = el2_ent_data.loc[(~el2_ent_data[exog].isna()) & (el2_ent_data['treat'] == 0) & (el2_ent_data['ent_type'] == 2), exog]
                _weights = el2_ent_data.loc[(~el2_ent_data[exog].isna()) & (el2_ent_data['treat'] == 0) & (el2_ent_data['ent_type'] == 2), 'ent_weight_el2'] 
                _amt = np.average(_var, weights=_weights) 
                
                ATE_string_spillover = ATE_string_spillover + ' + {}*{}*{}'.format(ent_type_weights[2], _amt, p)
    
            if 'C(ent_type)[3.0]:' in p:
                exog = p[len('C(ent_type)[3.0]:'):] 
                _var = el2_ent_data.loc[(~el2_ent_data[exog].isna()) & (el2_ent_data['treat'] == 1) & (el2_ent_data['ent_type'] == 3), exog]
                _weights = el2_ent_data.loc[(~el2_ent_data[exog].isna()) & (el2_ent_data['treat'] == 1) & (el2_ent_data['ent_type'] == 3), 'ent_weight_el2'] 
                _amt = np.average(_var, weights=_weights) 
                
                ATE_string_total = ATE_string_total + ' + {}*{}*{}'.format(ent_type_weights[3], _amt, p)
                
                _var = el2_ent_data.loc[(~el2_ent_data[exog].isna()) & (el2_ent_data['treat'] == 0) & (el2_ent_data['ent_type'] == 3), exog]
                _weights = el2_ent_data.loc[(~el2_ent_data[exog].isna()) & (el2_ent_data['treat'] == 0) & (el2_ent_data['ent_type'] == 3), 'ent_weight_el2'] 
                _amt = np.average(_var, weights=_weights) 
                
                ATE_string_spillover = ATE_string_spillover + ' + {}*{}*{}'.format(ent_type_weights[3], _amt, p)
                
    t_test_total = ge_gmm_results.t_test(ATE_string_total) 
    t_test_spillover = ge_gmm_results.t_test(ATE_string_spillover)
    results = {'b_total': t_test_total.effect[0], 'se_total': t_test_total.sd[0][0], 'p_total': t_test_total.pvalue,
               'b_spillover': t_test_spillover.effect[0], 'se_spillover': t_test_spillover.sd[0][0], 'p_spillover': t_test_spillover.pvalue}
    return results    

