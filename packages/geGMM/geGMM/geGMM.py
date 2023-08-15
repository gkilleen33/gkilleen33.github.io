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
            

class spatialGMM(GMM):
    """
    Custom GMM class that incorporates Conley SEs. Extends the statsmodels GMM class, and supports most functionality 
    from the package. 
    ----------
    latitude: A numpy array containing the latitude points for spatial SE calculations.
    longitude: A numpy array containing the longitude points for spatial SE calculations.
    bartlett: (Optional, default False) If False, uses a uniform distance kernel. If True uses Bartlett kernel. 
    distance_cutoff: (Optional, default 10) Distance cutoff for spatial SEs in km.
    """
    
    def __init__(self, endog, exog, instrument, latitude, longitude, k_moms=None, k_params=None,
                 missing='none', bartlett=False, distance_cutoff=10, **kwds):
        
        instrument = self._check_inputs(instrument, endog, latitude, longitude) # attaches if needed
        self.kernel = self._calc_kernel(latitude, longitude, distance_cutoff, bartlett=bartlett)  # Done once here so not recalculated on each iteration
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
    
    def _calc_kernel(self, lat, lon, cutoff=10, bartlett=False):
        K = np.empty([len(lat), len(lat)])
        for i in range(len(lat)):
            lati = lat.iloc[i]
            loni = lon.iloc[i]
            distancei = self._calc_distance(lati, loni, lat, lon)
            if bartlett:
                Ki = [d/cutoff if d <= cutoff else 0 for d in distancei]
            else:
                Ki = [1 if d <= cutoff else 0 for d in distancei]
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
    controls: (Optional) Patsy formula with controls to partial out 
    distance_cutoff: Default 10km: spatial distance cutoff for Conley error estimation in km
    bartlett: Default False, if True uses a Bartlett Kernel instead of uniform 
    parameters: A list of parameter names, as strings. Optional if simple IV case. Required if using other moments.

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

    Methods
    -------
    _partialOutControls(data, var_list, controls, weight_name=None)
        Partials out controls from each variable in var_list. Used internally.
    """
    
    def __init__(self, data, dep_vars, exog, instruments, moments=None, 
                 latitude='latitude', longitude='longitude', pweights=None, 
                 controls=None, distance_cutoff=10, bartlett=False, parameters=None, *args, **kwds):

        # If dep_vars specified as a string, convert to a list 
        if isinstance(dep_vars, str):
            dep_vars = dep_vars.split(',')
        
        # Get an initial list of variables to drop 
        if controls is not None:
            md = ModelDesc.from_formula('{} ~ {} + {} + {}'.format(dep_vars[0], exog, instruments, controls))
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
            vars_to_use.append(x.replace('C(', '').replace(')', ''))  # Handles categorical variables
        if pweights is not None:
            vars_to_use = vars_to_use + [pweights]
        vars_to_use = vars_to_use + ['latitude', 'longitude']
        vars_to_use = list(set(vars_to_use))  # Remove duplicates 
        
        _df = data.dropna(subset = vars_to_use)
        
        instruments = dmatrix(instruments + ' - 1', _df, return_type='dataframe')
        exog = dmatrix(exog + ' - 1', _df, return_type='dataframe')
    
        if controls is not None:
            control_df = dmatrix(controls, _df, return_type='dataframe')
            df_full = pd.concat([_df[dep_vars], _df[[latitude, longitude]], instruments, exog, control_df], axis=1)

        else:
            self.controls = None
            df_full = pd.concat([_df[dep_vars], _df[[latitude, longitude]], instruments, exog], axis=1)
            
        if pweights is not None:
            df_full = pd.concat([df_full, _df[pweights]], axis = 1)
            df_full.dropna(inplace=True)
            
            if controls:
                self.n_controls = len(control_df.columns)
                var_list = dep_vars + list(exog.columns) + list(instruments.columns)
                df_po = self._partialOutControls(data=df_full, var_list=var_list, controls=df_full[control_df.columns].values, weight_name=pweights)
            else:
                df_po = df_full 
                self.n_controls = 0 
        else:
            df_full.dropna(inplace=True)
            if controls:
                self.n_controls = len(control_df.columns)
                var_list = dep_vars + list(exog.columns) + list(instruments.columns)
                df_po = self._partialOutControls(data=df_full, var_list=var_list, controls=df_full[control_df.columns].values)
            else:
                self.n_controls = 0 
                df_po = df_full 
            
        self.instrument_names = list(instruments.columns)    
        instrument = df_po[instruments.columns].values
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
                                    bartlett=bartlett, distance_cutoff=distance_cutoff, *args, **kwds)
        
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
        else:
            weights = 1
            
        for var in var_list:
            # Get the dependent variable data 
            _dep = _df[var].values
    
            # Regress the dependent variable on the fixed effects variables.
            ind_reg = WLS(_dep, controls, weights=weights).fit()
            df_po[var] = ind_reg.resid
            
        return df_po
        

def calculate_optimal_radii(data, endog, dynamic_exog, dynamic_instruments, static_instruments=None, 
                            static_exog=None, static_controls=None, dynamic_controls=None, monotonic=True,
                            latitude='latitude', longitude='longitude', bartlett=False, distance_cutoff=10, pweights=None, maxiter=1,
                            optim_method='bfgs', optim_args=None):
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
        bartlett (default False): If True, uses a Bartlett kernel. If False uses a uniform kernel. 
        distance_cutoff (float default 10): Distance cutoff for kernel in km 
        pweights (optional, str): Name of column containing inverse probability weights in data 
        maxiter: (int default 1): Maximum number of GMM iterations. Default is 1, but if using 2 step GMM performance is better with set to 2.
        optim_method: (str, default 'bfgs') : scipy optimization method. bfgs is default for speed, 'nm' is more robust but slower. 
        optim_args: (optional, dict) : dictionary of arguments for the optimization algorithm (e.g. increase max iterations)

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
    
    for r in range(2, 22, 2):
        r2 = r - 2 
        if r == 2:
            add = 0 
            for inst in dynamic_instruments:
                if add == 0:
                    _dynamic_instrument_patsy = inst.format(r2, r)
                    add = 1 
                else:
                    _dynamic_instrument_patsy = _dynamic_instrument_patsy + ' + ' + inst.format(r2, r)
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
                        _dynamic_controls_patsy = c.format(r, r2)
                        add = 1 
                    else:
                        _dynamic_controls_patsy = _dynamic_controls_patsy + ' + ' + c.format(r2, r)
            
        else:
            if r > max_radius: # Stop execution and return the best performing value to that point 
                opt_r = min(bic_values, key=bic_values.get)
                return opt_r, exog_dict[opt_r], instruments_dict[opt_r], controls_dict[opt_r]   
            for inst in dynamic_instruments:
                _dynamic_instrument_patsy = _dynamic_instrument_patsy + ' + ' + inst.format(r2, r)
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
                     controls=_controls, distance_cutoff=distance_cutoff, bartlett=bartlett)
        
        _beta0 = np.ones(_model.n_exog)
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
        instruments_dict[r] = _exog

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
