from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import io
import base64
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')  # For server environments with no display
import matplotlib.pyplot as plt
import random
from typing import List, Dict, Any
from scipy.stats import chi2
from scipy.optimize import minimize

app = Flask(__name__)
app.secret_key = 'dev-key-change-this-in-production'  # needed for sessions

def four_pl(x, top, bottom, ic50, hill):
    """4-parameter logistic function"""
    x = np.maximum(x, 1e-10)
    ic50 = np.maximum(ic50, 1e-10)
    return bottom + (top - bottom) / (1 + (x/ic50)**hill)

def bootstrap_fit(concentrations: np.ndarray, responses: np.ndarray, n_iterations: int = 1000) -> Dict[str, List[float]]:
    """Perform bootstrap resampling to estimate parameter confidence intervals"""
    n_samples = len(concentrations)
    bootstrap_params = {
        'top': [], 'bottom': [], 'ic50': [], 'hill': []
    }
    
    for _ in range(n_iterations):
        try:
            indices = np.random.randint(0, n_samples, size=n_samples)
            boot_conc = concentrations[indices]
            boot_resp = responses[indices]
            
            # Fit curve for this bootstrap sample
            boot_results = fit_curve(boot_conc, boot_resp, skip_bootstrap=True)
            
            # Store the parameters
            for param in ['top', 'bottom', 'ic50', 'hill']:
                bootstrap_params[param].append(boot_results['parameters'][param])
            
        except Exception as e:
            print(f"Bootstrap iteration failed: {str(e)}")
            continue
            
    return bootstrap_params

def profile_likelihood_ci(concentrations: np.ndarray, responses: np.ndarray, 
                         best_params: Dict[str, float], param_name: str,
                         confidence: float = 0.95) -> tuple:
    """Calculate confidence interval using profile likelihood method"""
    
    # Critical chi-square value for desired confidence level
    crit_chi2 = chi2.ppf(confidence, df=1)
    best_nll = negative_log_likelihood(concentrations, responses, best_params)
    threshold = best_nll + crit_chi2/2
    
    # Log-transform IC50 for better numerical stability
    if param_name == 'ic50':
        best_value = np.log10(best_params[param_name])
    else:
        best_value = best_params[param_name]
    
    def objective(param_value):
        # Convert back from log space for IC50
        if param_name == 'ic50':
            actual_value = 10**param_value
        else:
            actual_value = param_value
            
        # Fix the parameter of interest and optimize others
        fixed_params = best_params.copy()
        fixed_params[param_name] = actual_value
        
        # Define constraints for other parameters
        bounds = {
            'top': (0, np.max(responses) * 2),
            'bottom': (0, np.max(responses)),
            'ic50': (1e-10, 1e5),
            'hill': (-20, 20)
        }
        del bounds[param_name]
        
        # Initial values closer to best fit
        x0 = [best_params[p] for p in bounds.keys()]
        
        # Optimize other parameters with multiple starting points if needed
        best_result = None
        best_fun = np.inf
        
        for attempt in range(3):  # Try a few different starting points
            try:
                result = minimize(
                    lambda x: profile_nll(x, concentrations, responses, 
                                        fixed_params, param_name, bounds),
                    x0=[v * (0.5 + random.random()) for v in x0],  # Randomize starting point
                    bounds=list(bounds.values()),
                    method='L-BFGS-B'
                )
                if result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result
            except:
                continue
                
        return best_fun - threshold if best_result else np.inf
    
    # Set appropriate bounds based on parameter
    if param_name == 'ic50':
        param_bounds = [(np.log10(1e-10), np.log10(1e5))]
    elif param_name == 'hill':
        param_bounds = [(-20, 20)]
    else:
        param_bounds = [(0, np.max(responses) * 2)]
    
    try:
        # Lower bound with multiple attempts
        lower = None
        for attempt in range(3):
            try:
                result = minimize(lambda x: -x[0],
                               x0=[best_value * (0.5 + 0.5 * random.random())],
                               bounds=param_bounds,
                               constraints={'type': 'ineq',
                                          'fun': lambda x: -objective(x[0])})
                if result.success:
                    lower = result.x[0]
                    break
            except:
                continue
        
        # Upper bound with multiple attempts
        upper = None
        for attempt in range(3):
            try:
                result = minimize(lambda x: x[0],
                               x0=[best_value * (1.5 + 0.5 * random.random())],
                               bounds=param_bounds,
                               constraints={'type': 'ineq',
                                          'fun': lambda x: -objective(x[0])})
                if result.success:
                    upper = result.x[0]
                    break
            except:
                continue
        
        # Convert back from log space for IC50
        if param_name == 'ic50' and lower is not None and upper is not None:
            lower = 10**lower
            upper = 10**upper
            
        return lower, upper
    except:
        return None, None

def negative_log_likelihood(concentrations: np.ndarray, responses: np.ndarray, 
                          params: Dict[str, float]) -> float:
    """Calculate negative log likelihood assuming normal errors"""
    predicted = four_pl(concentrations, 
                       params['top'], 
                       params['bottom'], 
                       params['ic50'], 
                       params['hill'])
    
    # Estimate variance
    residuals = responses - predicted
    variance = np.var(residuals)
    
    # Calculate negative log likelihood
    nll = 0.5 * len(responses) * np.log(2 * np.pi * variance)
    nll += 0.5 * np.sum(residuals**2) / variance
    return nll

def profile_nll(x, concentrations, responses, fixed_params, fixed_param, bounds):
    """Helper function for profile likelihood optimization"""
    # Reconstruct full parameter set
    params = fixed_params.copy()
    remaining_params = list(bounds.keys())
    for i, param in enumerate(remaining_params):
        params[param] = x[i]
    
    return negative_log_likelihood(concentrations, responses, params)

def fit_curve(concentrations, responses, skip_bootstrap=False, interpolate=False):
    try:
        if len(concentrations) < 4:
            raise ValueError("Need at least 4 concentrations for fitting")
        if np.any(~np.isfinite(responses)):
            raise ValueError("All responses must be finite numbers")
        
        # Convert inputs to numpy arrays
        concentrations = np.array(concentrations)
        responses = np.array(responses)
        
        # Calculate statistics for each concentration
        unique_conc = np.unique(concentrations)
        mean_responses = []
        std_responses = []
        percent_inhibition = []
        
        # Find control response (concentration = 0)
        control_mask = concentrations == 0
        if not np.any(control_mask):
            raise ValueError("No control (0 concentration) found in data")
        control_mean = np.mean(responses[control_mask])
        
        # Calculate normalized responses
        normalized_responses = (responses / control_mean) * 100
        
        for conc in unique_conc:
            mask = concentrations == conc
            mean_resp = np.mean(responses[mask])
            std_resp = np.std(responses[mask])
            mean_responses.append(mean_resp)
            std_responses.append(std_resp)
            
            inhibition = ((control_mean - mean_resp) / control_mean) * 100
            percent_inhibition.append(inhibition)
        
        mean_responses = np.array(mean_responses)
        percent_inhibition = np.array(percent_inhibition)
        
        if interpolate:
            # Use control as top and maximum inhibition as bottom
            top = 100  # normalized to 100%
            
            # Calculate mean response at each concentration
            unique_conc_responses = {}
            for conc in unique_conc:
                mask = concentrations == conc
                unique_conc_responses[conc] = np.mean(normalized_responses[mask])
            
            # Find the minimum mean response (maximum inhibition)
            # Exclude control (0 concentration)
            non_zero_conc = [c for c in unique_conc if c > 0]
            bottom = min(unique_conc_responses[c] for c in non_zero_conc)
            
            # Initial parameter guesses for IC50 and hill only
            p0 = [np.median(concentrations), 1.0]
            bounds = (
                [1e-10, 0.1],  # enforce positive hill slope
                [1e5, 20]
            )
            
            # Define interpolation fitting function
            def interpolate_fit(x, ic50, hill):
                return bottom + (top - bottom) / (1 + (x/ic50)**hill)
            
            # Fit the curve with fixed top and bottom
            popt, _ = curve_fit(interpolate_fit, 
                              concentrations, 
                              normalized_responses,
                              p0=p0,
                              bounds=bounds,
                              method='trf',
                              max_nfev=5000)
            
            # Pack results
            popt = [top, bottom, popt[0], popt[1]]
        else:
            # Original full parameter fitting
            p0 = [np.max(responses), np.min(responses), np.median(concentrations), 1.0]
            bounds = (
                [np.min(responses) * 0.5, 0, 1e-10, -20],
                [np.max(responses) * 2, np.max(responses), 1e5, 20]
            )
            
            popt, _ = curve_fit(four_pl, 
                              concentrations, 
                              responses,
                              p0=p0,
                              bounds=bounds,
                              method='trf',
                              max_nfev=5000)

        # Create results dictionary
        results = {
            'parameters': {
                'top': popt[0],
                'bottom': popt[1],
                'ic50': popt[2],
                'hill': popt[3]
            },
            'inhibition_data': {
                'concentrations': unique_conc.tolist(),
                'mean_responses': mean_responses.tolist(),
                'std_responses': std_responses,
                'percent_inhibition': percent_inhibition.tolist(),
                'fit_start_idx': 0
            },
            'interpolated': interpolate
        }
        
        if not skip_bootstrap:
            # Calculate R-squared for all points
            fitted_responses = four_pl(concentrations, *popt)
            residuals = responses - fitted_responses
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((responses - np.mean(responses))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate R-squared for means
            unique_conc = np.unique(concentrations)
            mean_responses_array = []
            fitted_means = []
            
            for conc in unique_conc:
                mask = concentrations == conc
                mean_responses_array.append(np.mean(responses[mask]))
                fitted_means.append(four_pl(conc, *popt))
                
            mean_responses_array = np.array(mean_responses_array)
            fitted_means = np.array(fitted_means)
            
            residuals_means = mean_responses_array - fitted_means
            ss_res_means = np.sum(residuals_means**2)
            ss_tot_means = np.sum((mean_responses_array - np.mean(mean_responses_array))**2)
            r_squared_means = 1 - (ss_res_means / ss_tot_means)
            
            results['diagnostics'] = {
                'r_squared': r_squared,
                'r_squared_means': r_squared_means
            }
        
        return results
        
    except Exception as e:
        raise ValueError(f"Error fitting curve: {str(e)}")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/confirm_data', methods=['POST'])
def confirm_data():
    raw_data = request.form.get('raw_data', '').strip()
    interpolate = request.form.get('interpolate') == 'true'
    fit_means = request.form.get('fit_means') == 'true'
    
    try:
        # Split into lines and filter out empty lines
        lines = [line.strip() for line in raw_data.split('\n') if line.strip()]
        
        data_rows = []
        for line in lines:
            # Split by tabs or multiple spaces
            parts = [p.strip() for p in line.split('\t') if p.strip()]
            if not parts:
                parts = [p.strip() for p in line.split() if p.strip()]
            
            if len(parts) < 2:
                raise ValueError(f"Invalid data format in line: {line}")
                
            # Convert all values to float
            row = [float(x) for x in parts]
            data_rows.append(row)
            
        if not data_rows:
            raise ValueError("No valid data found")
            
        # Store in session for next step
        session['data_rows'] = data_rows
        session['interpolate'] = interpolate
        session['fit_means'] = fit_means
        
        return render_template('confirm.html', 
                             data_rows=data_rows,
                             interpolate=interpolate,
                             fit_means=fit_means,
                             n_concentrations=len(data_rows),
                             n_replicates=len(data_rows[0])-1)
                             
    except Exception as e:
        return render_template('index.html', 
                             error=f"Error parsing data: {str(e)}")

@app.route('/fit', methods=['POST'])
def fit():
    # Get data from session
    data_rows = session.get('data_rows')
    interpolate = session.get('interpolate', False)
    fit_means = session.get('fit_means', False)
    
    if not data_rows:
        print("No data found in session")
        return redirect(url_for('index'))
    
    try:
        # Convert to numpy arrays
        data_array = np.array(data_rows)
        concentrations = data_array[:, 0]
        responses = data_array[:, 1:]
        
        # Find control response (zero concentration)
        control_mask = concentrations == 0
        if not np.any(control_mask):
            raise ValueError("No control (0 concentration) found in data")
            
        control_mean = np.mean(responses[control_mask])
        
        if fit_means:
            # Use means for fitting
            unique_conc = np.unique(concentrations)
            mean_responses_array = []
            
            for conc in unique_conc:
                mask = concentrations == conc
                mean_responses_array.append(np.mean(responses[mask]))
            
            # Normalize means
            mean_responses_array = np.array(mean_responses_array)
            normalized_means = (mean_responses_array / control_mean) * 100
            
            # Fit using means
            fit_results = fit_curve(unique_conc,
                                  normalized_means,
                                  skip_bootstrap=False,
                                  interpolate=interpolate)
        else:
            # Original code using all points
            flat_concentrations = np.repeat(concentrations, responses.shape[1])
            flat_responses = responses.flatten()
            normalized_responses = (flat_responses / control_mean) * 100
            
            fit_results = fit_curve(flat_concentrations, 
                                  normalized_responses,
                                  skip_bootstrap=False,
                                  interpolate=interpolate)
        
        # For plotting, calculate means and std of normalized responses
        mean_responses = np.mean(responses, axis=1) / control_mean * 100
        std_responses = np.std(responses, axis=1) / control_mean * 100
        
        # Print diagnostic information
        print("Fit diagnostics:")
        print(f"R-squared (all points): {fit_results['diagnostics']['r_squared']:.4f}")
        print(f"R-squared (means): {fit_results['diagnostics']['r_squared_means']:.4f}")
        
        # Generate points for the fitted curve
        x_fit = np.linspace(0, max(concentrations) + 100, 200)
        y_fit = four_pl(x_fit, 
                       fit_results['parameters']['top'],
                       fit_results['parameters']['bottom'],
                       fit_results['parameters']['ic50'],
                       fit_results['parameters']['hill'])
        
        # Create the plot
        plt.figure(figsize=(10, 6), dpi=100)
        plt.style.use('bmh')
        
        # Plot individual data points with some transparency
        for i in range(responses.shape[1]):
            plt.plot(concentrations, 
                    responses[:, i] / control_mean * 100,
                    'o', 
                    alpha=0.3,
                    markersize=4,
                    color='#2C3E50',
                    label='Individual replicates' if i == 0 else None)
        
        # Plot means with error bars
        plt.plot(concentrations,
                mean_responses,
                'o',
                markersize=8,
                color='#E74C3C',
                label='Mean')
        
        # Plot fitted curve
        plt.plot(x_fit, y_fit, '-', color='#E74C3C', linewidth=2, label='Fitted Curve')
        
        # Rest of the plotting code remains the same...
        plt.xlim(0, max(concentrations) + 100)
        tick_step = 100
        tick_locations = np.arange(0, max(concentrations) + 100 + tick_step, tick_step)
        plt.xticks(tick_locations)
        
        y_min = min(0, np.min(mean_responses - std_responses))
        y_max = max(100, np.max(mean_responses + std_responses))
        padding = (y_max - y_min) * 0.1
        plt.ylim(y_min - padding, y_max + padding)
        
        plt.xlabel('Concentration (μM)', fontsize=12, fontweight='bold')
        plt.ylabel('Response (%)', fontsize=12, fontweight='bold')
        plt.title('Dose-Response Curve', fontsize=14, fontweight='bold', pad=20)
        
        plt.axhline(y=100, color='gray', linestyle='--', alpha=0.2)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.2)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend(loc='best', framealpha=0.9, fontsize=10)
        plt.tight_layout()
        
        # Save plot
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
        img_buf.seek(0)
        plot_url = base64.b64encode(img_buf.read()).decode('ascii')
        plt.close()
        
        print("Created plot")
        
        return render_template('results.html',
                             parameters=fit_results['parameters'],
                             diagnostics=fit_results['diagnostics'],
                             plot_url=plot_url,
                             inhibition_data=fit_results['inhibition_data'])
                             
    except Exception as e:
        print(f"Exception in /fit route: {str(e)}")
        print(f"Exception type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return render_template('confirm.html',
                             error=f"Error during analysis: {str(e)}",
                             data_rows=data_rows,
                             n_concentrations=len(data_rows),
                             n_replicates=len(data_rows[0])-1)

@app.route('/update_curve', methods=['POST'])
def update_curve():
    try:
        params = request.json
        
        # Get data from session
        data_rows = session.get('data_rows')
        if not data_rows:
            return jsonify({'error': 'No data found'}), 400
            
        # Convert to numpy arrays
        data_array = np.array(data_rows)
        concentrations = data_array[:, 0]
        responses = data_array[:, 1:]
        
        # Find control response
        control_mask = concentrations == 0
        control_mean = np.mean(responses[control_mask])
        
        # Normalize responses
        flat_concentrations = np.repeat(concentrations, responses.shape[1])
        flat_responses = responses.flatten()
        normalized_responses = (flat_responses / control_mean) * 100
        
        # Calculate means and std of normalized responses for plotting
        mean_responses = np.mean(responses, axis=1) / control_mean * 100
        std_responses = np.std(responses, axis=1) / control_mean * 100
        
        # Generate new plot with updated parameters
        plt.figure(figsize=(10, 6), dpi=100)
        plt.style.use('bmh')
        
        # Plot individual points
        for i in range(responses.shape[1]):
            plt.plot(concentrations, 
                    responses[:, i] / control_mean * 100,
                    'o', 
                    alpha=0.3,
                    markersize=4,
                    color='#2C3E50',
                    label='Individual replicates' if i == 0 else None)
        
        # Plot means with error bars
        plt.plot(concentrations,
                mean_responses,
                'o',
                markersize=8,
                color='#E74C3C',
                label='Mean')
        
        # Plot updated fitted curve
        x_fit = np.linspace(0, max(concentrations) + 100, 200)
        y_fit = four_pl(x_fit, 
                       float(params['top']),
                       float(params['bottom']),
                       float(params['ic50']),
                       float(params['hill']))
        plt.plot(x_fit, y_fit, '-', color='#E74C3C', linewidth=2, label='Fitted Curve')
        
        # Rest of plotting code...
        plt.xlim(0, max(concentrations) + 100)
        plt.xlabel('Concentration (μM)', fontsize=12, fontweight='bold')
        plt.ylabel('Response (%)', fontsize=12, fontweight='bold')
        plt.title('Dose-Response Curve', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='best')
        plt.tight_layout()
        
        # Save and encode plot
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
        img_buf.seek(0)
        plot_url = base64.b64encode(img_buf.read()).decode('ascii')
        plt.close()
        
        return jsonify({'plot_url': plot_url})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/recalculate_stats', methods=['POST'])
def recalculate_stats():
    try:
        params = request.json
        
        # Get data from session
        data_rows = session.get('data_rows')
        if not data_rows:
            return jsonify({'error': 'No data found'}), 400
            
        # Convert to numpy arrays
        data_array = np.array(data_rows)
        concentrations = data_array[:, 0]
        responses = data_array[:, 1:]
        
        # Normalize responses
        control_mask = concentrations == 0
        control_mean = np.mean(responses[control_mask])
        flat_concentrations = np.repeat(concentrations, responses.shape[1])
        flat_responses = responses.flatten()
        normalized_responses = (flat_responses / control_mean) * 100
        
        # Create results with manual parameters
        results = {
            'parameters': {
                'top': float(params['top']),
                'bottom': float(params['bottom']),
                'ic50': float(params['ic50']),
                'hill': float(params['hill'])
            }
        }
        
        # Get confidence intervals through bootstrap
        bootstrap_params = bootstrap_fit(flat_concentrations, normalized_responses)
        
        # Calculate confidence intervals
        ci = {}
        for param in ['top', 'bottom', 'ic50', 'hill']:
            if bootstrap_params[param]:
                ci[param] = np.percentile(bootstrap_params[param], [2.5, 97.5])
                ci[param] = (ci[param][1] - ci[param][0]) / 2
            else:
                ci[param] = None
        
        # Calculate R-squared
        fitted_responses = four_pl(flat_concentrations, *[results['parameters'][p] for p in ['top', 'bottom', 'ic50', 'hill']])
        residuals = normalized_responses - fitted_responses
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((normalized_responses - np.mean(normalized_responses))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Generate new plot
        plot_url = generate_plot(concentrations, responses, control_mean, results['parameters'])
        
        return jsonify({
            'parameters': results['parameters'],
            'confidence_intervals': ci,
            'diagnostics': {
                'r_squared': r_squared,
                'bootstrap_iterations': len(bootstrap_params['top'])
            },
            'plot_url': plot_url
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def generate_plot(concentrations, responses, control_mean, parameters):
    # Create plot (same code as in update_curve)
    plt.figure(figsize=(10, 6), dpi=100)
    # ... (rest of plotting code)
    # Return base64 encoded plot
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
    img_buf.seek(0)
    plot_url = base64.b64encode(img_buf.read()).decode('ascii')
    plt.close()
    return plot_url

if __name__ == '__main__':
    app.run(debug=True) 