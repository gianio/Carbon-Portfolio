# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import datetime # Import datetime for date check
import pytz # For timezone handling
import traceback # For detailed error logging

# ==================================
# Refactored Data Loading Function (with BOM handling)
# ==================================
@st.cache_data
def load_and_prepare_data_simplified(uploaded_file):
    """
    Loads data from an uploaded CSV file, prepares and validates it.
    Handles potential UTF-8 BOM in CSV files.
    (Rest of docstring is the same as before)
    """
    try:
        # Added encoding='utf-8-sig' to handle potential BOM
        data = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        # Standardize column names to snake_case for internal processing
        data.columns = [str(col).lower().strip().replace(' ', '_') for col in data.columns]
    except Exception as read_error:
        return None, f"Error reading CSV file: {read_error}. Ensure it's a valid CSV.", [], [], []

    core_cols_std = ['project_name', 'project_type', 'priority']
    # This check happens *after* standardization
    missing_essential = [col for col in core_cols_std if col not in data.columns]

    if missing_essential:
        err_msg = (
            f"CSV missing essential columns after standardization. "
            f"Expected these standardized names: {core_cols_std}. "
            f"Standardized columns found in uploaded file: {list(data.columns)}. "
            f"The following essential columns were NOT found: {missing_essential}. "
            f"Please check your CSV file's header row. Common issues: typos, missing columns, or unexpected delimiters (this tool expects comma-separated values)."
        )
        return None, err_msg, [], [], []

    # Identify available years and numeric columns
    available_years = set()
    year_data_cols_found_std = []
    numeric_prefixes_std = ['price_', 'available_volume_']

    for col_std in data.columns:
        for prefix in numeric_prefixes_std:
            if col_std.startswith(prefix):
                year_part = col_std[len(prefix):]
                if year_part.isdigit() and len(year_part) == 4:
                    try:
                        year_val = int(year_part)
                        if 2000 <= year_val <= 2100:
                            available_years.add(year_val)
                            year_data_cols_found_std.append(col_std)
                    except ValueError:
                        pass
                    break

    if not available_years:
        err_msg = "No valid year columns found (e.g., 'price_2023', 'available_volume_2024'). Ensure columns follow 'prefix_YYYY' format with 4-digit years."
        # Return empty list for available_years but allow processing to continue if core columns are okay
        # This decision depends on whether year columns are strictly essential at this stage or can be handled later.
        # For now, let's allow it to proceed and let downstream logic handle lack of years if necessary.
        # However, the original script might implicitly require them.
        # If essential columns are fine, but no year columns, it's a different issue than missing core_cols_std.
        # Let's keep the original behavior: if no available_years, it's an error.
        return None, err_msg, [], [], [] # Original behavior: error out

    available_years = sorted(list(available_years))

    if 'priority' in data.columns:
        data['priority'] = pd.to_numeric(data['priority'], errors='coerce').fillna(0)

    for year in available_years:
        price_col_std = f"price_{year}"
        volume_col_std = f"available_volume_{year}"

        if price_col_std in year_data_cols_found_std and price_col_std in data.columns:
            data[price_col_std] = pd.to_numeric(data[price_col_std], errors='coerce')
            data[price_col_std] = data[price_col_std].fillna(0.0).clip(lower=0.0)
        
        if volume_col_std in year_data_cols_found_std and volume_col_std in data.columns:
            data[volume_col_std] = pd.to_numeric(data[volume_col_std], errors='coerce')
            data[volume_col_std] = data[volume_col_std].fillna(0).round().astype(int).clip(lower=0)

    invalid_types_found_list = []
    if 'project_type' in data.columns:
        data['project_type'] = data['project_type'].astype(str).str.lower().str.strip()
        valid_types = ['reduction', 'technical removal', 'natural removal']
        invalid_rows_mask = ~data['project_type'].isin(valid_types)
        if invalid_rows_mask.any():
            invalid_types_found_list = data.loc[invalid_rows_mask, 'project_type'].unique().tolist()
            data = data[~invalid_rows_mask].copy()
            if data.empty:
                return None, f"All rows had invalid project types or no valid project types left after filtering. Valid types: {', '.join(valid_types)}. Found: {', '.join(invalid_types_found_list)}", available_years, [], invalid_types_found_list
    else: # Should have been caught by the initial missing_essential check
        # This case should ideally not be reached if the first check is comprehensive.
        return None, "Critical error: 'project_type' column is unexpectedly missing after initial checks.", available_years, [], []


    cols_to_keep_std = core_cols_std[:]
    optional_cols_std = ['description', 'project_link']
    for col_std in optional_cols_std:
        if col_std in data.columns:
            cols_to_keep_std.append(col_std)

    cols_to_keep_std.extend([col for col in year_data_cols_found_std if col in data.columns])
    
    # Ensure all columns in cols_to_keep_std actually exist in data.columns to prevent KeyErrors
    final_cols_to_keep = [col for col in list(set(cols_to_keep_std)) if col in data.columns]
    data = data[final_cols_to_keep].copy()


    rename_map_to_desired = {
        'project_name': 'project name',
        'project_type': 'project type',
        'priority': 'priority',
        'description': 'Description',
        'project_link': 'Project Link'
    }
    for year_col_std in year_data_cols_found_std:
        if year_col_std in data.columns:
             rename_map_to_desired[year_col_std] = year_col_std.replace('_', ' ')

    data.rename(columns=rename_map_to_desired, inplace=True)

    project_names_list = []
    if 'project name' in data.columns:
        project_names_list = sorted(data['project name'].astype(str).unique().tolist())

    return data, None, available_years, project_names_list, invalid_types_found_list

# ==================================
# Configuration & Theming
# ==================================
st.set_page_config(layout="wide")
# --- Combined CSS ---
css = """
<style>
    /* ... CSS remains the same ... */
    /* Main App background - uncomment if desired */
    /* .stApp { background-color: #F1F8E9; } */

    /* --- Sidebar: Green Background & Text --- */
    [data-testid="stSidebar"] { background-color: #C8E6C9; padding-top: 2rem; } /* Lighter Green BG */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .st-bq {
        color: #1B5E20 !important; /* Dark Green Text/Labels */
    }
    /* --- Sidebar: Green Interactive Elements --- */
    [data-testid="stSidebar"] .stButton>button { background-color: #4CAF50; color: white; border: none; } /* Green Button */
    [data-testid="stSidebar"] .stButton>button:hover { background-color: #388E3C; color: white; } /* Darker Green Hover */
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div:nth-child(3) { background-color: #388E3C; } /* Dark Green slider handle range */
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] { background-color: #66BB6A; color: white; } /* Medium Green Tag */
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] span:first-child[aria-checked="true"] {
        background-color: #388E3C !important; /* Dark Green checkmark background */
    }
    /* --- Sidebar: Consistent Green Theme for other elements --- */
    [data-testid="stSidebar"] .stNumberInput input, [data-testid="stSidebar"] .stSelectbox select,
    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"],
    [data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] {
        border-color: #A5D6A7 !important; /* Light green border for widgets */
    }
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] { background: #A5D6A7; } /* Light green Ticks */
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] span:first-child {
        background-color: #A5D6A7 !important; /* Lighter green background for radio */
        border-color: #388E3C !important; /* Dark Green border */
    }

    /* --- Main Content --- */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; padding-left: 3rem; padding-right: 3rem; max-width: 100%; }
    h1, h2, h3, h4, h5, h6 { color: #1B5E20; } /* Titles green */
    .stDataFrame { border: 1px solid #A5D6A7; } /* Dataframe border green */

    /* Style for metric boxes */
    .metric-box {
        border: 2px solid #388E3C; /* Dark Green border */
        border-radius: 5px;
        padding: 10px; /* Adjusted padding */
        margin-bottom: 10px;
        background-color: #E8F5E9; /* Lightest Green background */
        text-align: center;
        font-size: 1.8em; /* Adjusted font size for value */
        /* Let columns handle width */
    }
    .metric-box b { /* Style for the label */
        display: block;
        margin-bottom: 5px;
        color: #1B5E20; /* Dark green label */
        font-size: 0.6em; /* Adjust label font size relative to value */
        font-weight: bold;
    }

    /* Experimental: Increase font size in dataframes */
    /* This CSS rule targets dataframe cells and headers. */
    /* It might be unstable across Streamlit versions. */
    .stDataFrame table td, .stDataFrame table th {
        font-size: 115%; /* Adjusted font size */
    }

    /* Styling for the download button */
    /* WARNING: Targeting specific Streamlit elements may break with updates */
    div[data-testid="stDownloadButton"] > button {
        background-color: #8ca734 !important; /* Requested Green */
        color: white !important;
        border: none !important; /* Remove border */
        padding: 0.8em 1.5em !important; /* Adjust padding for size */
        width: auto; /* Allow button to size based on content */
        font-size: 1.1em !important; /* Slightly larger font */
        font-weight: bold;
        border-radius: 5px; /* Add rounded corners */
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: #6b8e23 !important; /* Darker shade for hover */
        color: white !important;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Define Color Map for Charts (Green Theme)
type_color_map = {
    'technical removal': '#66BB6A', 'natural removal': '#AED581', 'reduction': '#388E3C'
}
default_color = '#BDBDBD' # Default color for unexpected types


# ==================================
# Allocation Function
# ==================================
# (Allocation function remains unchanged from your original code)
def allocate_portfolio(
    project_data: pd.DataFrame,
    selected_project_names: list,
    selected_years: list,
    start_year_portfolio: int,
    end_year_portfolio: int,
    constraint_type: str,
    annual_targets: dict,
    removal_target_percent_end_year: float,
    transition_speed: int,
    category_split: dict,
    favorite_project: str = None,
    priority_boost_percent: int = 10,
    min_target_fulfillment_percent: float = 0.95,
    min_allocation_chunk: int = 1 # Define minimum allocation unit
) -> tuple[dict, pd.DataFrame]:
    # ... Allocation function code remains the same ...
    portfolio_details = {year: [] for year in selected_years}
    yearly_summary_list = []

    # --- Initial Checks ---
    if not selected_project_names:
        st.warning("No projects selected for allocation.")
        return {}, pd.DataFrame(columns=['Year', f'Target {constraint_type}', 'Allocated Volume', 'Allocated Cost', 'Avg. Price', 'Actual Removal Vol %', 'Target Removal Vol %'])

    project_data_selected = project_data[project_data['project name'].isin(selected_project_names)].copy()

    if project_data_selected.empty:
        st.warning("Selected projects not found in the provided data.")
        return {}, pd.DataFrame(columns=['Year', f'Target {constraint_type}', 'Allocated Volume', 'Allocated Cost', 'Avg. Price', 'Actual Removal Vol %', 'Target Removal Vol %'])

    # --- Data Validation and Preparation ---
    required_base_cols = ['project name', 'project type', 'priority']
    price_cols_needed, volume_cols_needed = [], []
    for year in selected_years:
        price_cols_needed.append(f"price {year}")
        volume_cols_needed.append(f"available volume {year}")

    missing_base = [col for col in required_base_cols if col not in project_data_selected.columns]
    if missing_base:
        raise ValueError(f"Input data is missing required base columns: {', '.join(missing_base)}")

    missing_years_data = []
    for year_val in selected_years: # Renamed 'year' to 'year_val' to avoid conflict
        if f"price {year_val}" not in project_data_selected.columns:
            missing_years_data.append(f"price {year_val}")
        if f"available volume {year_val}" not in project_data_selected.columns:
            missing_years_data.append(f"available volume {year_val}")

    if missing_years_data:
        years_affected = sorted(list(set(int(col.split()[-1]) for col in missing_years_data if col.split()[-1].isdigit())))
        raise ValueError(f"Input data is missing price/volume information for required year(s): {', '.join(map(str, years_affected))}.")

    # Convert relevant columns to numeric, handle NaNs and negatives
    numeric_cols_to_check = ['priority'] + price_cols_needed + volume_cols_needed
    for col in numeric_cols_to_check:
        if col in project_data_selected.columns:
            project_data_selected[col] = pd.to_numeric(project_data_selected[col], errors='coerce')
            if col == 'priority':
                project_data_selected[col] = project_data_selected[col].fillna(0) # Default priority to 0 if missing
            elif col.startswith("available volume"):
                project_data_selected[col] = project_data_selected[col].fillna(0).apply(lambda x: int(x) if pd.notna(x) else 0).clip(lower=0)
            elif col.startswith("price"):
                project_data_selected[col] = project_data_selected[col].fillna(0.0).apply(lambda x: float(x) if pd.notna(x) else 0.0).clip(lower=0.0)

    all_project_types_in_selection = project_data_selected['project type'].unique()
    is_reduction_selected = 'reduction' in all_project_types_in_selection
    total_years_duration = end_year_portfolio - start_year_portfolio

    # --- Yearly Allocation Loop ---
    for year_val_loop in selected_years: # Renamed 'year' to 'year_val_loop'
        yearly_target = annual_targets.get(year_val_loop, 0) 
        price_col = f"price {year_val_loop}"
        volume_col = f"available volume {year_val_loop}"

        year_total_allocated_vol = 0
        year_total_allocated_cost = 0.0
        summary_template = {
            'Year': year_val_loop, f'Target {constraint_type}': yearly_target, 'Allocated Volume': 0,
            'Allocated Cost': 0.0, 'Avg. Price': 0.0, 'Actual Removal Vol %': 0.0,
            'Target Removal Vol %': 0.0 
        }

        if yearly_target <= 0:
            yearly_summary_list.append(summary_template); portfolio_details[year_val_loop] = []; continue

        target_percentages = {}
        if is_reduction_selected:
            start_removal_percent = 0.10; end_removal_percent = removal_target_percent_end_year
            progress = 1.0 if total_years_duration <= 0 else max(0, min(1, (year_val_loop - start_year_portfolio) / total_years_duration))
            exponent = 0.1 + (11 - transition_speed) * 0.2; progress_factor = progress ** exponent
            target_removal_percent_year = start_removal_percent + (end_removal_percent - start_removal_percent) * progress_factor
            min_removal_percent = min(start_removal_percent, end_removal_percent); max_removal_percent = max(start_removal_percent, end_removal_percent)
            target_removal_percent_year = max(min_removal_percent, min(max_removal_percent, target_removal_percent_year))
            tech_removal_pref = category_split.get('technical removal', 0); nat_removal_pref = category_split.get('natural removal', 0)
            total_removal_pref = tech_removal_pref + nat_removal_pref
            target_tech_removal = 0.0; target_nat_removal = 0.0
            if total_removal_pref > 1e-9:
                target_tech_removal = target_removal_percent_year * (tech_removal_pref / total_removal_pref)
                target_nat_removal = target_removal_percent_year * (nat_removal_pref / total_removal_pref)
            elif 'technical removal' in all_project_types_in_selection or 'natural removal' in all_project_types_in_selection:
                num_removal_types = ('technical removal' in all_project_types_in_selection) + ('natural removal' in all_project_types_in_selection)
                share = target_removal_percent_year / num_removal_types if num_removal_types > 0 else 0
                if 'technical removal' in all_project_types_in_selection: target_tech_removal = share
                if 'natural removal' in all_project_types_in_selection: target_nat_removal = share
            target_reduction = max(0.0, 1.0 - target_tech_removal - target_nat_removal)
            target_percentages = {'reduction': target_reduction, 'technical removal': target_tech_removal, 'natural removal': target_nat_removal}
        else:
            tech_removal_pref = category_split.get('technical removal', 0); nat_removal_pref = category_split.get('natural removal', 0)
            total_removal_pref = tech_removal_pref + nat_removal_pref
            tech_selected = 'technical removal' in all_project_types_in_selection; nat_selected = 'natural removal' in all_project_types_in_selection
            tech_alloc_share = 0.0; nat_alloc_share = 0.0
            if total_removal_pref > 1e-9:
                if tech_selected: tech_alloc_share = tech_removal_pref / total_removal_pref
                if nat_selected: nat_alloc_share = nat_removal_pref / total_removal_pref
            else:
                num_removal_types = tech_selected + nat_selected; share = 1.0 / num_removal_types if num_removal_types > 0 else 0
                if tech_selected: tech_alloc_share = share
                if nat_selected: nat_alloc_share = share
            total_alloc_share = tech_alloc_share + nat_alloc_share
            if total_alloc_share > 1e-9:
                target_percentages['technical removal'] = tech_alloc_share / total_alloc_share if tech_selected else 0.0
                target_percentages['natural removal'] = nat_alloc_share / total_alloc_share if nat_selected else 0.0
            else: target_percentages['technical removal'] = 0.0; target_percentages['natural removal'] = 0.0
            target_percentages['reduction'] = 0.0

        current_sum = sum(target_percentages.values())
        if abs(current_sum - 1.0) > 1e-6 and current_sum > 0:
            norm_factor = 1.0 / current_sum
            target_percentages = {ptype: share * norm_factor for ptype, share in target_percentages.items()}
        summary_template['Target Removal Vol %'] = (target_percentages.get('technical removal', 0) + target_percentages.get('natural removal', 0)) * 100

        projects_year_df = project_data_selected[(project_data_selected[price_col] > 0) & (project_data_selected[volume_col] >= min_allocation_chunk)].copy()
        projects_year_df['initial_allocated_volume'] = 0; projects_year_df['initial_allocated_cost'] = 0.0
        projects_year_df['final_priority'] = np.nan

        if projects_year_df.empty:
            yearly_summary_list.append(summary_template); portfolio_details[year_val_loop] = []; continue

        for project_type in all_project_types_in_selection:
            target_share = target_percentages.get(project_type, 0)
            if target_share <= 0: continue
            target_resource = yearly_target * target_share
            projects_of_type = projects_year_df[projects_year_df['project type'] == project_type].copy()
            if projects_of_type.empty: continue
            total_priority_in_type = projects_of_type['priority'].sum()
            if total_priority_in_type <= 0:
                num_projects_in_type = len(projects_of_type)
                projects_of_type['norm_prio_base'] = (1.0 / num_projects_in_type) if num_projects_in_type > 0 else 0
            else: projects_of_type['norm_prio_base'] = projects_of_type['priority'] / total_priority_in_type
            current_priorities = projects_of_type.set_index('project name')['norm_prio_base'].to_dict()
            final_priorities = current_priorities.copy()
            if favorite_project and favorite_project in final_priorities:
                fav_proj_base_prio = current_priorities[favorite_project]; boost_factor = priority_boost_percent / 100.0
                priority_increase = fav_proj_base_prio * boost_factor; new_fav_proj_prio = fav_proj_base_prio + priority_increase
                other_projects = [p for p in current_priorities if p != favorite_project]
                sum_other_priorities = sum(current_priorities[p] for p in other_projects)
                temp_priorities = {favorite_project: new_fav_proj_prio}; reduction_factor = 0
                if sum_other_priorities > 1e-9: reduction_factor = priority_increase / sum_other_priorities
                for name in other_projects: temp_priorities[name] = max(0, current_priorities[name] * (1 - reduction_factor))
                total_final_prio = sum(temp_priorities.values())
                if total_final_prio > 1e-9: final_priorities = {p: prio / total_final_prio for p, prio in temp_priorities.items()}
                elif favorite_project in temp_priorities: final_priorities = {favorite_project: 1.0}
            project_weights = {}; total_weight = 0
            if constraint_type == 'Budget':
                for _, row in projects_of_type.iterrows():
                    name = row['project name']; final_prio = final_priorities.get(name, 0); price = row[price_col]
                    weight = final_prio * price if price > 0 else 0; project_weights[name] = weight; total_weight += weight
            for idx, row in projects_of_type.iterrows():
                name = row['project name']; final_prio = final_priorities.get(name, 0); available_vol = row[volume_col]; price = row[price_col]
                allocated_volume = 0; allocated_cost = 0.0
                projects_year_df.loc[projects_year_df['project name'] == name, 'final_priority'] = final_prio
                if final_prio <= 0 or price <= 0 or available_vol < min_allocation_chunk: continue
                if constraint_type == 'Volume':
                    target_volume_proj = target_resource * final_prio; allocated_volume = min(target_volume_proj, available_vol)
                elif constraint_type == 'Budget':
                    if total_weight > 1e-9:
                        weight_normalized = project_weights.get(name, 0) / total_weight; target_budget_proj = target_resource * weight_normalized
                        target_volume_proj = target_budget_proj / price if price > 0 else 0
                        allocated_volume = min(target_volume_proj, available_vol)
                    else: allocated_volume = 0
                allocated_volume = int(max(0, math.floor(allocated_volume)))
                if allocated_volume >= min_allocation_chunk:
                    allocated_cost = allocated_volume * price
                    projects_year_df.loc[projects_year_df['project name'] == name, 'initial_allocated_volume'] += allocated_volume
                    projects_year_df.loc[projects_year_df['project name'] == name, 'initial_allocated_cost'] += allocated_cost
                    year_total_allocated_vol += allocated_volume; year_total_allocated_cost += allocated_cost

        target_threshold = yearly_target * min_target_fulfillment_percent
        current_metric_total = year_total_allocated_cost if constraint_type == 'Budget' else year_total_allocated_vol
        if current_metric_total < target_threshold and yearly_target > 0:
            needed = target_threshold - current_metric_total
            projects_year_df['remaining_volume'] = projects_year_df[volume_col] - projects_year_df['initial_allocated_volume']
            adjustment_candidates = projects_year_df[(projects_year_df['remaining_volume'] >= min_allocation_chunk) & (projects_year_df[price_col] > 0)].sort_values(by='priority', ascending=False).copy()
            for idx, row in adjustment_candidates.iterrows():
                if needed <= 1e-6: break
                name = row['project name']; price = row[price_col]; available_for_adj = row['remaining_volume']
                # volume_to_add = 0; cost_to_add = 0.0 # Declared earlier
                if constraint_type == 'Volume': add_vol = int(math.floor(min(available_for_adj, needed)))
                else: max_affordable_vol = int(math.floor(needed / price)) if price > 0 else 0; add_vol = min(available_for_adj, max_affordable_vol)
                
                volume_to_add_current = 0 # Use a temporary variable for this iteration
                cost_to_add_current = 0.0

                if add_vol >= min_allocation_chunk:
                    cost_increase = add_vol * price
                    if constraint_type == 'Volume' or cost_increase <= needed * 1.1 or cost_increase < price * min_allocation_chunk * 1.5 :
                        volume_to_add_current = add_vol; cost_to_add_current = cost_increase
                        needed -= cost_to_add_current if constraint_type == 'Budget' else volume_to_add_current
                if volume_to_add_current > 0:
                    projects_year_df.loc[idx, 'initial_allocated_volume'] += volume_to_add_current
                    projects_year_df.loc[idx, 'initial_allocated_cost'] += cost_to_add_current
                    projects_year_df.loc[idx, 'remaining_volume'] -= volume_to_add_current
                    year_total_allocated_vol += volume_to_add_current; year_total_allocated_cost += cost_to_add_current

        final_allocations_list = []
        final_year_allocations_df = projects_year_df[projects_year_df['initial_allocated_volume'] >= min_allocation_chunk].copy()
        for idx, row in final_year_allocations_df.iterrows():
            current_price = row.get(price_col, None)
            final_allocations_list.append({'project name': row['project name'], 'type': row['project type'], 'allocated_volume': row['initial_allocated_volume'], 'allocated_cost': row['initial_allocated_cost'], 'price_used': current_price, 'priority_applied': row['final_priority']})
        portfolio_details[year_val_loop] = final_allocations_list # Use year_val_loop
        summary_template['Allocated Volume'] = year_total_allocated_vol; summary_template['Allocated Cost'] = year_total_allocated_cost
        summary_template['Avg. Price'] = (year_total_allocated_cost / year_total_allocated_vol) if year_total_allocated_vol > 0 else 0.0
        removal_volume = sum(p['allocated_volume'] for p in final_allocations_list if p['type'] in ['technical removal', 'natural removal'])
        summary_template['Actual Removal Vol %'] = (removal_volume / year_total_allocated_vol * 100) if year_total_allocated_vol > 0 else 0.0
        yearly_summary_list.append(summary_template)

    yearly_summary_df = pd.DataFrame(yearly_summary_list)

    if constraint_type == 'Budget':
        check_df = yearly_summary_df.copy()
        check_df['Target Budget'] = check_df['Year'].map(annual_targets).fillna(0)
        is_overbudget = check_df['Allocated Cost'] > check_df['Target Budget'] * 1.001
        overbudget_years_df = check_df[is_overbudget]
        if not overbudget_years_df.empty:
            st.warning(f"Budget target may have been slightly exceeded in year(s): {overbudget_years_df['Year'].tolist()} due to allocation adjustments or minimum chunk requirements.")

    return portfolio_details, yearly_summary_df


# ==================================
# Streamlit App Layout & Logic
# ==================================
# --- Sidebar ---
# (Sidebar code remains largely the same, ensure it calls the updated load_and_prepare_data_simplified)
with st.sidebar:
    st.markdown("## 1. Load Data")
    df_upload = st.file_uploader("Upload Project Data CSV", type="csv", key="uploader_sidebar", help="CSV required columns: `project name`, `project type` ('reduction', 'technical removal', 'natural removal'), `priority`. Also needs `price_YYYY` and `available_volume_YYYY` columns for relevant years. Optional: `description`, `project_link`.")
    default_values = {'working_data_full': None, 'selected_years': [], 'selected_projects': [], 'project_names': [], 'favorite_projects_selection': [], 'actual_start_year': None, 'actual_end_year': None, 'available_years_in_data': [], 'constraint_type': 'Volume', 'removal_target_end_year': 0.8, 'transition_speed': 5, 'category_split': {'technical removal': 0.5, 'natural removal': 0.5}, 'annual_targets': {}, 'master_target': None, 'data_loaded_successfully': False, 'years_slider_sidebar': 5, 'min_fulfillment_perc': 95, 'removal_preference_slider': 5, 'min_alloc_chunk': 1}
    for key, default_value in default_values.items():
        if key not in st.session_state: st.session_state[key] = default_value

    if df_upload:
        try:
            data, error_msg, available_years_in_data, project_names_list, invalid_types_found = load_and_prepare_data_simplified(df_upload) # Using the updated function

            if invalid_types_found:
                st.sidebar.warning(f"Ignored rows with invalid project types: {', '.join(invalid_types_found)}. Valid types are 'reduction', 'technical removal', 'natural removal'.")
            if error_msg:
                st.sidebar.error(error_msg) # This will now show the more detailed error
                st.session_state.data_loaded_successfully = False
                st.session_state.working_data_full = None
                st.session_state.project_names = []
                st.session_state.available_years_in_data = []
                st.session_state.selected_projects = []
                st.session_state.annual_targets = {}
            else:
                st.session_state.project_names = project_names_list
                st.session_state.available_years_in_data = available_years_in_data
                st.session_state.working_data_full = data
                st.session_state.data_loaded_successfully = True
                st.sidebar.success("Data loaded successfully!")

                current_selection = st.session_state.get('selected_projects', [])
                valid_current_selection = [p for p in current_selection if p in project_names_list]
                if not valid_current_selection and project_names_list:
                    st.session_state.selected_projects = project_names_list 
                else:
                    st.session_state.selected_projects = valid_current_selection
                st.session_state.annual_targets = {} 
        except Exception as e:
            st.sidebar.error(f"An unexpected error occurred during file processing: {e}")
            st.sidebar.error(f"Traceback: {traceback.format_exc()}")
            st.session_state.data_loaded_successfully = False; st.session_state.working_data_full = None; st.session_state.project_names = []; st.session_state.available_years_in_data = []; st.session_state.selected_projects = []; st.session_state.annual_targets = {}

# ... The rest of the Streamlit app layout and logic remains the same ...
# --- Settings (Displayed only if data loaded successfully) ---
    if st.session_state.get('data_loaded_successfully', False):
        data_for_ui = st.session_state.working_data_full
        available_years_in_data = st.session_state.available_years_in_data
        project_names_list = st.session_state.project_names

        if not available_years_in_data: # This check is important
            st.sidebar.warning("No usable year data (e.g. price_YYYY, available_volume_YYYY columns) found in the uploaded file, or selected range has no data.")
            # Reset year-dependent states if no years available
            st.session_state.selected_years = []
            st.session_state.actual_start_year = None
            st.session_state.actual_end_year = None

        else: # Only proceed if available_years_in_data is not empty
            st.markdown("## 2. Portfolio Settings")
            min_year_data = min(available_years_in_data)
            max_year_data = max(available_years_in_data)
            max_years_slider = min(20, max(1, max_year_data - min_year_data + 1)) 

            current_slider_val = st.session_state.get('years_slider_sidebar', 5)
            if current_slider_val > max_years_slider : current_slider_val = max_years_slider
            if current_slider_val < 1: current_slider_val = 1


            years_to_plan = st.slider(
                f"Years to Plan (Starting {min_year_data})",
                1, max_years_slider,
                value=current_slider_val, 
                key='years_slider_sidebar_widget' 
            )
            st.session_state.years_slider_sidebar = years_to_plan 

            start_year_selected = min_year_data
            end_year_selected = start_year_selected + years_to_plan - 1

            actual_years_present_in_data = []
            if data_for_ui is not None: 
                for year_val_iter in range(start_year_selected, end_year_selected + 1): # Renamed year to year_val_iter
                    price_col = f"price {year_val_iter}" 
                    vol_col = f"available volume {year_val_iter}"
                    if price_col in data_for_ui.columns and vol_col in data_for_ui.columns:
                        actual_years_present_in_data.append(year_val_iter)

            st.session_state.selected_years = actual_years_present_in_data

            if not st.session_state.selected_years:
                st.sidebar.error(f"No data available for selected period ({start_year_selected}-{end_year_selected}). Adjust 'Years to Plan' slider or check CSV data columns.")
                st.session_state.actual_start_year = None
                st.session_state.actual_end_year = None
            else:
                st.session_state.actual_start_year = min(st.session_state.selected_years)
                st.session_state.actual_end_year = max(st.session_state.selected_years)
                st.sidebar.markdown(f"Planning Horizon: **{st.session_state.actual_start_year} - {st.session_state.actual_end_year}**")

                st.session_state.constraint_type = st.radio(
                    "Constraint Type:",
                    ('Volume', 'Budget'),
                    index=['Volume', 'Budget'].index(st.session_state.get('constraint_type', 'Volume')),
                    key='constraint_type_sidebar_widget', 
                    horizontal=True,
                    help="Choose whether annual targets are defined in tons (Volume) or currency (Budget)."
                )
                constraint_type = st.session_state.constraint_type 

                st.markdown("### Annual Target Settings")
                master_target_value = st.session_state.get('master_target')

                if constraint_type == 'Volume':
                    default_val_vol = 1000
                    if master_target_value is not None and isinstance(master_target_value, (int, float)):
                        try: default_val_vol = int(float(master_target_value))
                        except (ValueError, TypeError): pass
                    default_target = st.number_input(
                        "Default Annual Target Volume (t):", min_value=0, step=100,
                        value=default_val_vol, key='master_volume_sidebar',
                        help="Set a default target volume per year. You can override specific years below."
                    )
                else: 
                    default_val_bud = 100000.0
                    if master_target_value is not None and isinstance(master_target_value, (int, float)):
                        try: default_val_bud = float(master_target_value)
                        except (ValueError, TypeError): pass
                    default_target = st.number_input(
                        "Default Annual Target Budget (€):", min_value=0.0, step=1000.0,
                        value=default_val_bud, format="%.2f", key='master_budget_sidebar',
                        help="Set a default target budget per year. You can override specific years below."
                    )
                st.session_state.master_target = default_target

                with st.expander("Customize Annual Targets (+/-)", expanded=False):
                    current_annual_targets = st.session_state.get('annual_targets', {})
                    updated_targets_from_inputs = {}

                    if not st.session_state.selected_years:
                        st.caption("Select years using the 'Years to Plan' slider above first.")
                    else:
                        for year_val_iter_2 in st.session_state.selected_years: # Renamed year to year_val_iter_2
                            year_target_value = current_annual_targets.get(year_val_iter_2, default_target)
                            input_key = f"target_{year_val_iter_2}_{constraint_type}" 
                            label = f"Target {year_val_iter_2} [t]" if constraint_type == 'Volume' else f"Target {year_val_iter_2} [€]"

                            if constraint_type == 'Volume':
                                try: input_val = int(year_target_value)
                                except (ValueError, TypeError): input_val = int(default_target)
                                updated_targets_from_inputs[year_val_iter_2] = st.number_input(
                                    label, min_value=0, step=100, value=input_val, key=input_key
                                )
                            else: 
                                try: input_val = float(year_target_value)
                                except (ValueError, TypeError): input_val = float(default_target)
                                updated_targets_from_inputs[year_val_iter_2] = st.number_input(
                                    label, min_value=0.0, step=1000.0, value=input_val, format="%.2f", key=input_key
                                )
                    st.session_state.annual_targets = updated_targets_from_inputs

                st.sidebar.markdown("### Allocation Goal & Preferences")
                min_fulfill = st.sidebar.slider(f"Min. Target Fulfillment (%)", 50, 100, st.session_state.get('min_fulfillment_perc', 95), help=f"Attempt >= this % of target {constraint_type} via adjustment.", key='min_fulfill_perc_sidebar')
                st.session_state.min_fulfillment_perc = min_fulfill
                min_chunk_val = st.sidebar.number_input("Min. Allocation Unit (t)", 1, step=1, value=st.session_state.get('min_alloc_chunk', 1), help="Smallest amount (tons) to allocate per project/year.", key='min_alloc_chunk_sidebar')
                st.session_state.min_alloc_chunk = int(min_chunk_val)

                st.sidebar.markdown("### Removal Volume Transition (If 'Reduction' Projects Used)")
                reduction_present = False
                if data_for_ui is not None and 'project type' in data_for_ui.columns: 
                    reduction_present = 'reduction' in data_for_ui['project type'].unique()

                if reduction_present: st.sidebar.info("Transition settings apply if 'Reduction' projects selected.")
                else: st.sidebar.info("Transition inactive: No 'Reduction' projects found or selected.")

                removal_help = f"Target % vol from Removals in final year ({st.session_state.actual_end_year if st.session_state.actual_end_year else 'N/A'}). Guides mix if 'Reduction' selected."
                rem_target_slider = st.sidebar.slider(f"Target Removal Vol % ({st.session_state.actual_end_year if st.session_state.actual_end_year else 'N/A'})", 0, 100, int(st.session_state.get('removal_target_end_year', 0.8)*100), help=removal_help, key='removal_perc_slider_sidebar', disabled=not reduction_present or not st.session_state.actual_end_year)
                st.session_state.removal_target_end_year = rem_target_slider / 100.0
                st.session_state.transition_speed = st.sidebar.slider("Transition Speed", 1, 10, st.session_state.get('transition_speed', 5), help="Ramp-up speed (1=Slow, 10=Fast) if Reductions selected.", key='transition_speed_slider_sidebar', disabled=not reduction_present)

                st.sidebar.markdown("### Removal Category Preference")
                removal_types_present = False
                if data_for_ui is not None and 'project type' in data_for_ui.columns:
                    removal_types_present = any(pt in data_for_ui['project type'].unique() for pt in ['technical removal', 'natural removal'])

                rem_pref_val = st.sidebar.slider("Technical vs Natural Preference", 1, 10, st.session_state.get('removal_preference_slider', 5), format="%d", key='removal_pref_slider_sidebar', help="1 leans Natural, 5 balanced, 10 leans Technical.", disabled=not removal_types_present)
                st.session_state['removal_preference_slider'] = rem_pref_val
                tech_pref_ratio = (rem_pref_val - 1) / 9.0
                st.session_state.category_split = {'technical removal': tech_pref_ratio, 'natural removal': 1.0 - tech_pref_ratio}

                st.sidebar.markdown("## 3. Select Projects")
                if not project_names_list:
                    st.sidebar.warning("No projects available from loaded data.")
                else:
                    current_selected_projects = st.session_state.get('selected_projects', [])
                    valid_defaults = [p for p in current_selected_projects if p in project_names_list]
                    if not valid_defaults and project_names_list: 
                        valid_defaults = project_names_list

                    st.session_state.selected_projects = st.sidebar.multiselect(
                        "Select projects to include:",
                        options=project_names_list,
                        default=valid_defaults,
                        key='project_selector_sidebar' 
                    )

                    if data_for_ui is not None and 'priority' in data_for_ui.columns: 
                        boost_options = [p for p in project_names_list if p in st.session_state.selected_projects]
                        if boost_options:
                            current_favorite = st.session_state.get('favorite_projects_selection', [])
                            valid_default_favorite = [f for f in current_favorite if f in boost_options][:1]
                            st.session_state.favorite_projects_selection = st.sidebar.multiselect(
                                "Favorite Project (Priority Boost):",
                                options=boost_options, default=valid_default_favorite,
                                key='favorite_selector_sidebar', 
                                max_selections=1, help="Boost priority for one project."
                            )
                        else:
                            st.sidebar.info("Select projects first to enable boost.")
                            st.session_state.favorite_projects_selection = [] 
                    else:
                        st.sidebar.info("Boost disabled: No 'priority' column in data.")
                        st.session_state.favorite_projects_selection = [] 


# ==================================
# Main Page Content
# ==================================
st.markdown(f"<h1 style='color: #8ca734;'>Carbon Portfolio Builder</h1>", unsafe_allow_html=True)
st.markdown("---")

if not st.session_state.get('data_loaded_successfully', False):
    st.info("👋 Welcome! Please upload your project data CSV using the sidebar menu to get started.")

elif st.session_state.get('data_loaded_successfully', False):
    st.markdown("## Project Offerings") 
    st.caption("To get more information on the specific project please unfold the list and click the link to the project slide.") 

    project_df_display = st.session_state.working_data_full.copy() 
    available_years_main = st.session_state.available_years_in_data # Renamed to avoid conflict
    price_cols_available = [f"price {year_val_main}" for year_val_main in available_years_main if f"price {year_val_main}" in project_df_display.columns] # Renamed year

    if price_cols_available: 
        project_df_display['Average Price'] = project_df_display[price_cols_available].mean(axis=1, skipna=True)
        project_df_display['Average Price'] = project_df_display['Average Price'].fillna(0.0)
    else:
        project_df_display['Average Price'] = 0.0 

    with st.expander("View Project Details"):
        if not project_df_display.empty:
            display_cols_options = ['project name', 'project type', 'Average Price'] 
            column_config = {
                    "Average Price": st.column_config.NumberColumn("Avg Price (€/t)", help="Average price across all years available in the input data.", format="€%.2f")
            }
            if 'Project Link' in project_df_display.columns: 
                display_cols_options.append('Project Link')
                column_config["Project Link"] = st.column_config.LinkColumn("Project Link", display_text="Visit ->", help="Click link to visit project page (if available)")

            st.dataframe(project_df_display[display_cols_options], column_config=column_config, hide_index=True, use_container_width=True)
        else:
            st.write("Project data is loaded but appears to be empty or no projects with valid types were found.")
    st.markdown("---") 


if st.session_state.get('data_loaded_successfully', False):
    if not st.session_state.get('selected_projects'):
        st.warning("⚠️ Please select the projects you want to include in the portfolio using the sidebar (Section 3).")
    elif not st.session_state.get('selected_years'): # Check selected_years, not just available_years_in_data
        st.warning("⚠️ No valid years identified for the selected planning horizon, or no data available for those years. Please adjust the 'Years to Plan' slider (Section 2) or verify the year columns in your CSV data.")
    else:
        required_keys = ['working_data_full', 'selected_projects', 'selected_years','actual_start_year', 'actual_end_year', 'constraint_type', 'annual_targets', 'removal_target_end_year', 'transition_speed', 'category_split', 'favorite_projects_selection', 'min_fulfillment_perc', 'min_alloc_chunk', 'master_target']
        keys_present = all(
            k in st.session_state and
            (st.session_state.get(k) is not None if k not in ['annual_targets', 'favorite_projects_selection', 'working_data_full', 'master_target'] else True) 
            for k in required_keys
        )
        keys_present = keys_present and ('annual_targets' in st.session_state) and ('favorite_projects_selection' in st.session_state) and st.session_state.working_data_full is not None


        if keys_present:
            try:
                fav_proj = st.session_state.favorite_projects_selection[0] if st.session_state.favorite_projects_selection else None
                constraint = st.session_state.constraint_type
                min_chunk = st.session_state.min_alloc_chunk
                annual_targets_to_use = st.session_state.annual_targets.copy() # Use a copy

                if not annual_targets_to_use and st.session_state.selected_years and st.session_state.master_target is not None : 
                    annual_targets_to_use = {year_val_calc: st.session_state.master_target for year_val_calc in st.session_state.selected_years} # Renamed year


                if constraint == 'Budget': st.info(f"**Budget Mode:** Projects get budget via weighted priority/price. May get 0 vol if budget < cost of {min_chunk}t. Adjustment step may add later.")
                st.success(f"**Allocation Goal:** Attempting ≥ **{st.session_state.min_fulfillment_perc}%** of annual target {constraint}.")

                with st.spinner("Calculating portfolio allocation... Please wait."):
                    results, summary = allocate_portfolio(
                        project_data=st.session_state.working_data_full,
                        selected_project_names=st.session_state.selected_projects,
                        selected_years=st.session_state.selected_years,
                        start_year_portfolio=st.session_state.actual_start_year,
                        end_year_portfolio=st.session_state.actual_end_year,
                        constraint_type=constraint,
                        annual_targets=annual_targets_to_use, 
                        removal_target_percent_end_year=st.session_state.removal_target_end_year,
                        transition_speed=st.session_state.transition_speed,
                        category_split=st.session_state.category_split,
                        favorite_project=fav_proj,
                        priority_boost_percent=10, 
                        min_target_fulfillment_percent=st.session_state.min_fulfillment_perc / 100.0,
                        min_allocation_chunk=min_chunk
                    )

                details_list = []
                if results:
                    for year_v, projects_v in results.items():
                        if projects_v: 
                            for proj_v in projects_v:
                                if isinstance(proj_v, dict) and (proj_v.get('allocated_volume', 0) >= min_chunk or proj_v.get('allocated_cost', 0) > 1e-6):
                                    details_list.append({
                                        'year': year_v,
                                        'project name': proj_v.get('project name'),
                                        'type': proj_v.get('type'),
                                        'volume': proj_v.get('allocated_volume', 0),
                                        'price': proj_v.get('price_used', None), 
                                        'cost': proj_v.get('allocated_cost', 0.0)
                                    })
                details_df = pd.DataFrame(details_list)
                pivot_display = pd.DataFrame() 

                st.markdown("## Portfolio Summary")
                col_l, col_r = st.columns([2, 1.2], gap="large")

                with col_l:
                    st.markdown("#### Key Metrics (Overall)")
                    if not summary.empty:
                        total_cost_all_years = summary['Allocated Cost'].sum()
                        total_volume_all_years = summary['Allocated Volume'].sum()
                        overall_avg_price = total_cost_all_years / total_volume_all_years if total_volume_all_years > 0 else 0.0
                        st.markdown(f"""<div class="metric-box"><b>Total Portfolio Cost</b> € {total_cost_all_years:,.2f}</div>""", unsafe_allow_html=True)
                        st.markdown(f"""<div class="metric-box"><b>Total Portfolio Volume</b> {total_volume_all_years:,.0f} t</div>""", unsafe_allow_html=True)
                        st.markdown(f"""<div class="metric-box"><b>Overall Average Price</b> € {overall_avg_price:,.2f} /t</div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="metric-box"><b>Total Portfolio Cost</b> - </div>""", unsafe_allow_html=True)
                        st.markdown(f"""<div class="metric-box"><b>Total Portfolio Volume</b> - </div>""", unsafe_allow_html=True)
                        st.markdown(f"""<div class="metric-box"><b>Overall Average Price</b> - </div>""", unsafe_allow_html=True)
                        st.warning("Could not calculate overall portfolio metrics. Summary table is empty.")

                with col_r:
                    st.markdown("#### Volume by Project Type")
                    if not details_df.empty:
                        pie_data = details_df.groupby('type')['volume'].sum().reset_index()
                        pie_data = pie_data[pie_data['volume'] > 1e-6] 
                        if not pie_data.empty:
                            st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
                            fig_pie = px.pie(pie_data, values='volume', names='type', color='type', color_discrete_map=type_color_map)
                            fig_pie.update_layout(
                                showlegend=True, legend_title_text='Project Type',
                                legend_orientation="h", legend_yanchor="bottom", legend_y=-0.2,
                                legend_xanchor="center", legend_x=0.5,
                                margin=dict(t=5, b=50, l=0, r=0), height=350
                            )
                            fig_pie.update_traces(textposition='inside', textinfo='percent', sort=False, hole=.3, marker=dict(line=dict(color='#FFFFFF', width=1)))
                            st.plotly_chart(fig_pie, use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        elif not details_df.empty and pie_data.empty : 
                            st.caption("No significant volume allocated to display in pie chart.")
                        # else: # This case is less likely if details_df was empty and handled below
                             # st.caption("Could not generate pie chart data from allocation details.")
                    else:
                        st.caption("No detailed allocation data available for pie chart.")

                st.markdown("---") 

                if details_df.empty and summary.empty:
                    st.warning("No allocation data was generated. Cannot display plots or tables.")
                elif details_df.empty : 
                    st.warning("No detailed project allocations available for the composition plot or detailed table.")
                else: 
                    st.markdown("### Portfolio Composition & Price Over Time")
                    summary_plot_data = details_df.groupby(['year', 'type']).agg(
                        volume=('volume', 'sum'),
                        cost=('cost', 'sum')
                    ).reset_index()

                    price_summary_data = pd.DataFrame()
                    if not summary.empty:
                        price_summary_data = summary[['Year', 'Avg. Price']].rename(columns={'Year':'year', 'Avg. Price':'avg_price'})

                    fig_composition = make_subplots(specs=[[{"secondary_y": True}]])
                    y_metric = 'volume' if constraint == 'Volume' else 'cost'
                    y_label = 'Allocated Volume (t)' if constraint == 'Volume' else 'Allocated Cost (€)'
                    y_format = '{:,.0f}' if constraint == 'Volume' else '€{:,.2f}'
                    y_hover_label = 'Volume' if constraint == 'Volume' else 'Cost'
                    type_order = ['reduction', 'natural removal', 'technical removal'] 
                    types_in_results = summary_plot_data['type'].unique()

                    for t_name in type_order:
                        if t_name in types_in_results:
                            df_type = summary_plot_data[summary_plot_data['type'] == t_name]
                            if not df_type.empty and y_metric in df_type.columns and df_type[y_metric].sum() > 1e-6:
                                fig_composition.add_trace(
                                    go.Bar(
                                        x=df_type['year'], y=df_type[y_metric], name=t_name.replace('_', ' ').capitalize(),
                                        marker_color=type_color_map.get(t_name, default_color),
                                        hovertemplate=f'Year: %{{x}}<br>Type: {t_name.replace("_", " ").capitalize()}<br>{y_hover_label}: %{{y:{y_format}}}<extra></extra>'
                                    ), secondary_y=False
                                )

                    if not price_summary_data.empty and 'avg_price' in price_summary_data.columns: # check column exists
                        fig_composition.add_trace(
                            go.Scatter(
                                x=price_summary_data['year'], y=price_summary_data['avg_price'], name='Avg Price (€/t)',
                                mode='lines+markers', marker=dict(symbol='circle', size=8), line=dict(color='#1B5E20', width=3),
                                hovertemplate='Year: %{x}<br>Avg Price: €%{y:,.2f}/t<extra></extra>'
                            ), secondary_y=True
                        )
                    if not summary.empty and 'Actual Removal Vol %' in summary.columns:
                        fig_composition.add_trace(
                            go.Scatter(
                                x=summary['Year'], y=summary['Actual Removal Vol %'], name='Actual Removal Vol %',
                                mode='lines+markers', line=dict(color='darkorange', dash='dash'), marker=dict(symbol='star', size=8),
                                hovertemplate='Year: %{x}<br>Actual Removal: %{y:.1f}%<extra></extra>'
                            ), secondary_y=True
                        )

                    y2_max_limit = 105 
                    if not price_summary_data.empty and 'avg_price' in price_summary_data.columns and not price_summary_data['avg_price'].empty:
                        y2_max_limit = max(y2_max_limit, price_summary_data['avg_price'].max(skipna=True) * 1.1)
                    if not summary.empty and 'Actual Removal Vol %' in summary.columns and not summary['Actual Removal Vol %'].empty:
                        y2_max_limit = max(y2_max_limit, summary['Actual Removal Vol %'].max(skipna=True) * 1.1)


                    fig_composition.update_layout(
                        xaxis_title='Year', yaxis_title=y_label, yaxis2_title='Avg Price (€/t) / Actual Removal %',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        barmode='stack', template="plotly_white", margin=dict(t=20, l=0, r=0, b=0),
                        yaxis=dict(rangemode='tozero'),
                        yaxis2=dict(rangemode='tozero', range=[0, y2_max_limit if pd.notna(y2_max_limit) and y2_max_limit > 0 else 105]), 
                        hovermode="x unified"
                    )
                    if st.session_state.selected_years:
                        fig_composition.update_xaxes(tickmode='array', tickvals=st.session_state.selected_years, dtick=1 if len(st.session_state.selected_years) < 20 else math.ceil(len(st.session_state.selected_years)/10) ) # Adjust dtick for many years
                    st.plotly_chart(fig_composition, use_container_width=True)

                st.markdown("### Detailed Allocation by Project and Year")
                if not details_df.empty or not summary.empty: 
                    try:
                        pivot_final = pd.DataFrame()
                        years_present_in_results_table = []
                        if not details_df.empty:
                            years_present_in_results_table = sorted(details_df['year'].unique())
                        elif not summary.empty:
                             years_present_in_results_table = sorted(summary['Year'].unique())
                        else: # Should not happen if outer if caught it
                            years_present_in_results_table = st.session_state.selected_years


                        if not details_df.empty:
                            pivot_intermediate = pd.pivot_table(details_df,
                                                                values=['volume', 'cost', 'price'],
                                                                index=['project name', 'type'],
                                                                columns='year',
                                                                aggfunc={'volume': 'sum', 'cost': 'sum', 'price': 'first'}) 

                            if not pivot_intermediate.empty:
                                pivot_final = pivot_intermediate.swaplevel(0, 1, axis=1) 
                                metric_order_table = ['volume', 'cost', 'price'] # Renamed metric_order

                                final_multi_index_cols_table = pd.MultiIndex.from_product( # Renamed final_multi_index
                                    [years_present_in_results_table, metric_order_table],
                                    names=['year', 'metric']
                                )
                                pivot_final = pivot_final.reindex(columns=final_multi_index_cols_table).fillna(0) 
                                pivot_final = pivot_final.sort_index(axis=1, level=[0, 1]) 
                                pivot_final.index.names = ['Project Name', 'Type']
                        
                        total_data_dict_table = {} # Renamed total_data_dict
                        if not summary.empty:
                            summary_indexed = summary.set_index('Year')
                            for year_val_table in years_present_in_results_table: # Renamed year_val
                                if year_val_table in summary_indexed.index:
                                    vol = summary_indexed.loc[year_val_table, 'Allocated Volume']
                                    cost_val = summary_indexed.loc[year_val_table, 'Allocated Cost']
                                    avg_price_val = summary_indexed.loc[year_val_table, 'Avg. Price']
                                else: 
                                    vol, cost_val, avg_price_val = 0, 0.0, 0.0
                                total_data_dict_table[(year_val_table, 'volume')] = vol
                                total_data_dict_table[(year_val_table, 'cost')] = cost_val
                                total_data_dict_table[(year_val_table, 'price')] = avg_price_val 

                        total_row_index_table = pd.MultiIndex.from_tuples([('Total Portfolio', 'All Types')], names=['Project Name', 'Type']) # Renamed total_row_index
                        total_row_df_table = pd.DataFrame(total_data_dict_table, index=total_row_index_table) # Renamed total_row_df
                        
                        if not pivot_final.empty:
                             total_row_df_table = total_row_df_table.reindex(columns=pivot_final.columns).fillna(0)
                        elif not total_row_df_table.empty and years_present_in_results_table: # If only total_row_df exists, sort its columns
                            metric_order_table_2 = ['volume', 'cost', 'price'] # Renamed
                            total_row_df_table = total_row_df_table.reindex(columns=pd.MultiIndex.from_product(
                                [years_present_in_results_table, metric_order_table_2], names=['year', 'metric'])).fillna(0)
                            total_row_df_table = total_row_df_table.sort_index(axis=1, level=[0,1])


                        if pivot_final.empty and total_row_df_table.empty:
                            st.info("No detailed allocation data to display in table.")
                            pivot_display = pd.DataFrame() 
                        elif pivot_final.empty:
                            pivot_display = total_row_df_table
                        else: 
                            pivot_display = pd.concat([pivot_final, total_row_df_table])

                        if not pivot_display.empty:
                            pivot_display = pivot_display.fillna(0) 
                            formatter_table = {} # Renamed formatter
                            for year_col_table, metric_col_table in pivot_display.columns: # Renamed
                                if metric_col_table == 'volume': formatter_table[(year_col_table, metric_col_table)] = '{:,.0f} t'
                                elif metric_col_table == 'cost': formatter_table[(year_col_table, metric_col_table)] = '€{:,.2f}'
                                elif metric_col_table == 'price':
                                    formatter_table[(year_col_table, metric_col_table)] = lambda x: f'€{x:,.2f}/t' if pd.notna(x) and x != 0 else ('-' if x==0 and metric_col_table == 'price' else ('€0.00/t' if x==0 else '-'))


                            st.dataframe(pivot_display.style.format(formatter_table, na_rep="-"), use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Could not create detailed allocation table: {e}")
                        st.error(f"Traceback: {traceback.format_exc()}")
                        pivot_display = pd.DataFrame() 
                else:
                    st.info("No allocation details or summary data to display in the table.")
                    pivot_display = pd.DataFrame() 

                if not pivot_display.empty:
                    csv_df_download = pivot_display.copy() # Renamed csv_df
                    csv_df_download.columns = [f"{int(col[0]) if pd.notna(col[0]) else 'UnknownYear'}_{col[1]}" for col in csv_df_download.columns.values] # Handle potential NaN in year
                    csv_df_download = csv_df_download.reset_index() 
                    try:
                        csv_string_download = csv_df_download.to_csv(index=False, encoding='utf-8') # Renamed csv_string
                        st.markdown("---")
                        st.download_button(
                            label="Download Detailed Allocation (CSV)",
                            data=csv_string_download,
                            file_name=f"portfolio_allocation_{datetime.date.today()}.csv",
                            mime='text/csv',
                            key='download-csv'
                        )
                    except Exception as e:
                        st.error(f"Error preparing CSV for download: {e}")


            except ValueError as e:
                st.error(f"Configuration or Allocation Error: {e}")
            except KeyError as e:
                st.error(f"Data Error: Missing key: '{e}'. This might be due to an unexpected CSV column name or a mismatch in selected data vs. available data. Please check your CSV and selections.")
            except Exception as e:
                st.error(f"An unexpected error occurred in the main calculation/display logic: {e}")
                st.error(f"Traceback: {traceback.format_exc()}")
        else:
            st.error("⚠️ Critical settings are missing or invalid. Please ensure data is loaded and all sidebar settings under 'Portfolio Settings' are configured correctly. You might need to re-upload the data or adjust the sliders.")

# --- Footer ---
st.markdown("---")
try:
    zurich_tz = pytz.timezone('Europe/Zurich')
    now_zurich = datetime.datetime.now(zurich_tz)
    st.caption(f"Report generated: {now_zurich.strftime('%Y-%m-%d %H:%M:%S %Z')}")
except Exception as e:
    now_local = datetime.datetime.now()
    st.caption(f"Report generated: {now_local.strftime('%Y-%m-%d %H:%M:%S')} (Timezone: Server Local)")
