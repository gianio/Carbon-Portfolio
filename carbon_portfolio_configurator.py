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
# from io import StringIO # Not needed

# ==================================
# Configuration & Theming (Green Sidebar, Custom Controls)
# ==================================
st.set_page_config(layout="wide")
# --- Combined CSS ---
# Note: Using data-testid can be brittle and break with Streamlit updates.
css = """
<style>
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
        font-size: 115%; /* Adjusted from 150% for potentially better fit */
    }

    /* Styling for the download button */
    /* WARNING: Targeting specific Streamlit elements may break with updates */
    /* This targets the button directly now, as it's no longer in nested columns */
    div[data-testid="stDownloadButton"] > button {
        background-color: #8ca734 !important; /* Requested Green */
        color: white !important;
        border: none !important; /* Remove border */
        padding: 0.8em 1.5em !important; /* Adjust padding for size */
        /* width: 100%; REMOVED - let button size naturally or set specific width if needed */
        width: auto; /* Allow button to size based on content */
        font-size: 1.1em !important; /* Slightly larger font */
        font-weight: bold;
        border-radius: 5px; /* Add rounded corners */
        /* display: inline-block; */ /* May help alignment if needed */
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
# (Allocation function remains unchanged from the previous version)
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
    """
    Allocates a portfolio of carbon projects based on user-defined constraints and preferences.
    (Docstring details omitted for brevity - see previous version if needed)
    """
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
    for year in selected_years:
        if f"price {year}" not in project_data_selected.columns:
            missing_years_data.append(f"price {year}")
        if f"available volume {year}" not in project_data_selected.columns:
            missing_years_data.append(f"available volume {year}")

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
                 # Fill NaN with 0, convert to int, ensure non-negative
                 project_data_selected[col] = project_data_selected[col].fillna(0).apply(lambda x: int(x) if pd.notna(x) else 0).clip(lower=0)
             elif col.startswith("price"):
                 # Fill NaN with 0.0, convert to float, ensure non-negative
                 project_data_selected[col] = project_data_selected[col].fillna(0.0).apply(lambda x: float(x) if pd.notna(x) else 0.0).clip(lower=0.0)

    all_project_types_in_selection = project_data_selected['project type'].unique()
    is_reduction_selected = 'reduction' in all_project_types_in_selection
    total_years_duration = end_year_portfolio - start_year_portfolio

    # --- Yearly Allocation Loop ---
    for year in selected_years:
        yearly_target = annual_targets.get(year, 0) # Get target for the year, default 0
        price_col = f"price {year}"
        volume_col = f"available volume {year}"

        year_total_allocated_vol = 0
        year_total_allocated_cost = 0.0
        summary_template = {
            'Year': year,
            f'Target {constraint_type}': yearly_target,
            'Allocated Volume': 0,
            'Allocated Cost': 0.0,
            'Avg. Price': 0.0,
            'Actual Removal Vol %': 0.0,
            'Target Removal Vol %': 0.0 # Will be calculated below
        }

        # Skip year if target is zero or negative
        if yearly_target <= 0:
            yearly_summary_list.append(summary_template)
            portfolio_details[year] = []
            continue

        # --- Calculate Target Percentages per Project Type for the Year ---
        target_percentages = {} # {project_type: target_share}
        # (Logic for calculating target_percentages remains the same)
        # ... (target percentage calculation logic as before) ...
        if is_reduction_selected:
            start_removal_percent = 0.10 # Minimum removal target at start
            end_removal_percent = removal_target_percent_end_year
            progress = 1.0 if total_years_duration <= 0 else max(0, min(1, (year - start_year_portfolio) / total_years_duration))
            exponent = 0.1 + (11 - transition_speed) * 0.2
            progress_factor = progress ** exponent
            target_removal_percent_year = start_removal_percent + (end_removal_percent - start_removal_percent) * progress_factor
            min_removal_percent = min(start_removal_percent, end_removal_percent)
            max_removal_percent = max(start_removal_percent, end_removal_percent)
            target_removal_percent_year = max(min_removal_percent, min(max_removal_percent, target_removal_percent_year))
            tech_removal_pref = category_split.get('technical removal', 0)
            nat_removal_pref = category_split.get('natural removal', 0)
            total_removal_pref = tech_removal_pref + nat_removal_pref
            target_tech_removal = 0.0
            target_nat_removal = 0.0
            if total_removal_pref > 1e-9: # Avoid division by zero
                 target_tech_removal = target_removal_percent_year * (tech_removal_pref / total_removal_pref)
                 target_nat_removal = target_removal_percent_year * (nat_removal_pref / total_removal_pref)
            elif 'technical removal' in all_project_types_in_selection or 'natural removal' in all_project_types_in_selection:
                num_removal_types = ('technical removal' in all_project_types_in_selection) + ('natural removal' in all_project_types_in_selection)
                share = target_removal_percent_year / num_removal_types if num_removal_types > 0 else 0
                if 'technical removal' in all_project_types_in_selection: target_tech_removal = share
                if 'natural removal' in all_project_types_in_selection: target_nat_removal = share
            target_reduction = max(0.0, 1.0 - target_tech_removal - target_nat_removal)
            target_percentages = {
                'reduction': target_reduction,
                'technical removal': target_tech_removal,
                'natural removal': target_nat_removal
            }
        else: # Only removal projects selected (or only one type)
            tech_removal_pref = category_split.get('technical removal', 0)
            nat_removal_pref = category_split.get('natural removal', 0)
            total_removal_pref = tech_removal_pref + nat_removal_pref
            tech_selected = 'technical removal' in all_project_types_in_selection
            nat_selected = 'natural removal' in all_project_types_in_selection
            tech_alloc_share = 0.0
            nat_alloc_share = 0.0
            if total_removal_pref > 1e-9: # Use preferences if set
                if tech_selected: tech_alloc_share = tech_removal_pref / total_removal_pref
                if nat_selected: nat_alloc_share = nat_removal_pref / total_removal_pref
            else: # If no preference, split equally among selected removal types
                 num_removal_types = tech_selected + nat_selected
                 share = 1.0 / num_removal_types if num_removal_types > 0 else 0
                 if tech_selected: tech_alloc_share = share
                 if nat_selected: nat_alloc_share = share
            total_alloc_share = tech_alloc_share + nat_alloc_share
            if total_alloc_share > 1e-9:
                 target_percentages['technical removal'] = tech_alloc_share / total_alloc_share if tech_selected else 0.0
                 target_percentages['natural removal'] = nat_alloc_share / total_alloc_share if nat_selected else 0.0
            else: # Handle case where neither type might be selected
                 target_percentages['technical removal'] = 0.0
                 target_percentages['natural removal'] = 0.0
            target_percentages['reduction'] = 0.0 # No reduction target if no reduction projects

        # Normalize target percentages to sum to 1.0
        current_sum = sum(target_percentages.values())
        if abs(current_sum - 1.0) > 1e-6 and current_sum > 0:
            norm_factor = 1.0 / current_sum
            target_percentages = {ptype: share * norm_factor for ptype, share in target_percentages.items()}

        summary_template['Target Removal Vol %'] = (target_percentages.get('technical removal', 0) + target_percentages.get('natural removal', 0)) * 100

        # --- Filter Projects for the Current Year ---
        projects_year_df = project_data_selected[
            (project_data_selected[price_col] > 0) &
            (project_data_selected[volume_col] >= min_allocation_chunk)
        ].copy()
        projects_year_df['initial_allocated_volume'] = 0
        projects_year_df['initial_allocated_cost'] = 0.0
        projects_year_df['final_priority'] = np.nan # Store the priority used after boost

        if projects_year_df.empty:
            yearly_summary_list.append(summary_template)
            portfolio_details[year] = []
            continue

        # --- Initial Allocation based on Priority and Targets ---
        for project_type in all_project_types_in_selection:
            # (Logic for initial allocation remains the same)
            # ... (initial allocation logic as before) ...
            target_share = target_percentages.get(project_type, 0)
            if target_share <= 0: continue # Skip if no target share for this type
            target_resource = yearly_target * target_share # Target volume or budget for this type
            projects_of_type = projects_year_df[projects_year_df['project type'] == project_type].copy()
            if projects_of_type.empty: continue # No projects of this type available this year
            total_priority_in_type = projects_of_type['priority'].sum()
            if total_priority_in_type <= 0: # If no priorities set, distribute equally
                num_projects_in_type = len(projects_of_type)
                projects_of_type['norm_prio_base'] = (1.0 / num_projects_in_type) if num_projects_in_type > 0 else 0
            else:
                projects_of_type['norm_prio_base'] = projects_of_type['priority'] / total_priority_in_type
            current_priorities = projects_of_type.set_index('project name')['norm_prio_base'].to_dict()
            final_priorities = current_priorities.copy() # Start with base priorities
            if favorite_project and favorite_project in final_priorities:
                fav_proj_base_prio = current_priorities[favorite_project]
                boost_factor = priority_boost_percent / 100.0
                priority_increase = fav_proj_base_prio * boost_factor
                new_fav_proj_prio = fav_proj_base_prio + priority_increase
                other_projects = [p for p in current_priorities if p != favorite_project]
                sum_other_priorities = sum(current_priorities[p] for p in other_projects)
                temp_priorities = {favorite_project: new_fav_proj_prio}
                reduction_factor = 0
                if sum_other_priorities > 1e-9: # Calculate reduction needed for others
                     reduction_factor = priority_increase / sum_other_priorities
                for name in other_projects:
                     temp_priorities[name] = max(0, current_priorities[name] * (1 - reduction_factor))
                total_final_prio = sum(temp_priorities.values())
                if total_final_prio > 1e-9:
                    final_priorities = {p: prio / total_final_prio for p, prio in temp_priorities.items()}
                elif favorite_project in temp_priorities: # Edge case: only favorite has priority
                    final_priorities = {favorite_project: 1.0}
            project_weights = {}; total_weight = 0
            if constraint_type == 'Budget':
                for _, row in projects_of_type.iterrows():
                    name = row['project name']; final_prio = final_priorities.get(name, 0)
                    price = row[price_col]; weight = final_prio * price if price > 0 else 0
                    project_weights[name] = weight; total_weight += weight
            for idx, row in projects_of_type.iterrows():
                name = row['project name']; final_prio = final_priorities.get(name, 0)
                available_vol = row[volume_col]; price = row[price_col]
                allocated_volume = 0; allocated_cost = 0.0
                projects_year_df.loc[projects_year_df['project name'] == name, 'final_priority'] = final_prio
                if final_prio <= 0 or price <= 0 or available_vol < min_allocation_chunk: continue
                if constraint_type == 'Volume':
                    target_volume_proj = target_resource * final_prio
                    allocated_volume = min(target_volume_proj, available_vol)
                elif constraint_type == 'Budget':
                    if total_weight > 1e-9:
                        weight_normalized = project_weights.get(name, 0) / total_weight
                        target_budget_proj = target_resource * weight_normalized
                        target_volume_proj = target_budget_proj / price if price > 0 else 0 # Added check for price > 0
                        allocated_volume = min(target_volume_proj, available_vol)
                    else: allocated_volume = 0
                allocated_volume = int(max(0, math.floor(allocated_volume)))
                if allocated_volume >= min_allocation_chunk:
                    allocated_cost = allocated_volume * price
                    projects_year_df.loc[projects_year_df['project name'] == name, 'initial_allocated_volume'] += allocated_volume
                    projects_year_df.loc[projects_year_df['project name'] == name, 'initial_allocated_cost'] += allocated_cost
                    year_total_allocated_vol += allocated_volume
                    year_total_allocated_cost += allocated_cost


        # --- Adjustment Step to Meet Minimum Fulfillment ---
        target_threshold = yearly_target * min_target_fulfillment_percent
        current_metric_total = year_total_allocated_cost if constraint_type == 'Budget' else year_total_allocated_vol

        if current_metric_total < target_threshold and yearly_target > 0:
            # (Logic for adjustment step remains the same)
            # ... (adjustment step logic as before) ...
            needed = target_threshold - current_metric_total # Amount of volume/budget still needed
            projects_year_df['remaining_volume'] = projects_year_df[volume_col] - projects_year_df['initial_allocated_volume']
            adjustment_candidates = projects_year_df[
                (projects_year_df['remaining_volume'] >= min_allocation_chunk) &
                (projects_year_df[price_col] > 0)
            ].sort_values(by='priority', ascending=False).copy() # Use original priority for fairness
            for idx, row in adjustment_candidates.iterrows():
                if needed <= 1e-6: break # Stop if threshold met
                name = row['project name']; price = row[price_col]; available_for_adj = row['remaining_volume']
                volume_to_add = 0; cost_to_add = 0.0
                if constraint_type == 'Volume':
                    add_vol = int(math.floor(min(available_for_adj, needed)))
                else: # Budget constraint
                    max_affordable_vol = int(math.floor(needed / price)) if price > 0 else 0
                    add_vol = min(available_for_adj, max_affordable_vol)
                if add_vol >= min_allocation_chunk:
                    cost_increase = add_vol * price
                    if constraint_type == 'Volume' or cost_increase <= needed * 1.1 or cost_increase < price * min_allocation_chunk * 1.5 :
                         volume_to_add = add_vol; cost_to_add = cost_increase
                         needed -= cost_to_add if constraint_type == 'Budget' else volume_to_add
                if volume_to_add > 0:
                    projects_year_df.loc[idx, 'initial_allocated_volume'] += volume_to_add
                    projects_year_df.loc[idx, 'initial_allocated_cost'] += cost_to_add
                    projects_year_df.loc[idx, 'remaining_volume'] -= volume_to_add # Keep track of remaining
                    year_total_allocated_vol += volume_to_add
                    year_total_allocated_cost += cost_to_add


        # --- Finalize Year Results ---
        final_allocations_list = []
        final_year_allocations_df = projects_year_df[projects_year_df['initial_allocated_volume'] >= min_allocation_chunk].copy()

        for idx, row in final_year_allocations_df.iterrows():
            current_price = row.get(price_col, None) # Price used in this year's allocation
            final_allocations_list.append({
                'project name': row['project name'],
                'type': row['project type'],
                'allocated_volume': row['initial_allocated_volume'],
                'allocated_cost': row['initial_allocated_cost'],
                'price_used': current_price,
                'priority_applied': row['final_priority'] # Store the normalized/boosted priority
            })

        portfolio_details[year] = final_allocations_list

        # Update summary template with final figures for the year
        summary_template['Allocated Volume'] = year_total_allocated_vol
        summary_template['Allocated Cost'] = year_total_allocated_cost
        summary_template['Avg. Price'] = (year_total_allocated_cost / year_total_allocated_vol) if year_total_allocated_vol > 0 else 0.0
        # Calculate actual removal volume percentage
        removal_volume = sum(p['allocated_volume'] for p in final_allocations_list if p['type'] in ['technical removal', 'natural removal'])
        summary_template['Actual Removal Vol %'] = (removal_volume / year_total_allocated_vol * 100) if year_total_allocated_vol > 0 else 0.0

        yearly_summary_list.append(summary_template)
    # --- End Yearly Allocation Loop ---

    yearly_summary_df = pd.DataFrame(yearly_summary_list)

    # --- Optional: Budget Check Warning ---
    if constraint_type == 'Budget':
        check_df = yearly_summary_df.copy()
        check_df['Target Budget'] = check_df['Year'].map(annual_targets).fillna(0)
        is_overbudget = check_df['Allocated Cost'] > check_df['Target Budget'] * 1.001
        overbudget_years_df = check_df[is_overbudget]
        if not overbudget_years_df.empty:
            st.warning(f"Budget target may have been slightly exceeded in year(s): {overbudget_years_df['Year'].tolist()} due to allocation adjustments or minimum chunk requirements.")
    # --- End Budget Check ---

    return portfolio_details, yearly_summary_df


# ==================================
# Streamlit App Layout & Logic
# ==================================

# --- Sidebar ---
# (Sidebar code remains unchanged from the previous version)
with st.sidebar:
    st.markdown("## 1. Load Data")
    df_upload = st.file_uploader(
        "Upload Project Data CSV",
        type="csv",
        key="uploader_sidebar",
        help="CSV required columns: `project name`, `project type` ('reduction', 'technical removal', 'natural removal'), `priority`. Also needs `price_YYYY` and `available_volume_YYYY` columns for relevant years. Optional: `description`, `project_link`."
        )

    # --- Initialize Session State ---
    default_values = {
        'working_data_full': None, 'selected_years': [], 'selected_projects': [], 'project_names': [],
        'favorite_projects_selection': [], 'actual_start_year': None, 'actual_end_year': None,
        'available_years_in_data': [], 'constraint_type': 'Volume', 'removal_target_end_year': 0.8,
        'transition_speed': 5, 'category_split': {'technical removal': 0.5, 'natural removal': 0.5},
        'annual_targets': {}, 'master_target': None, 'data_loaded_successfully': False,
        'years_slider_sidebar': 5, 'min_fulfillment_perc': 95, 'removal_preference_slider': 5,
        'min_alloc_chunk': 1
    }
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # --- Load and Prepare Data ---
    if df_upload:
        @st.cache_data
        def load_and_prepare_data(uploaded_file):
            """Loads, validates, standardizes, and prepares the uploaded CSV data."""
            try:
                data = pd.read_csv(uploaded_file)
                data.columns = data.columns.str.lower().str.strip().str.replace(' ', '_')
            except Exception as read_error:
                return None, f"Error reading CSV file: {read_error}", [], [], []
            core_cols_std = ['project_name', 'project_type', 'priority']
            optional_cols_std = ['description', 'project_link']
            missing_essential = [col for col in core_cols_std if col not in data.columns]
            if missing_essential:
                return None, f"CSV is missing essential columns: {', '.join(missing_essential)}", [], [], []
            numeric_prefixes_std = ['price_', 'available_volume_']
            cols_to_convert_numeric = ['priority']; available_years = set(); year_data_cols_found = []
            for col in data.columns:
                for prefix in numeric_prefixes_std:
                    year_part = col[len(prefix):]
                    if col.startswith(prefix) and year_part.isdigit():
                        cols_to_convert_numeric.append(col); year_data_cols_found.append(col)
                        available_years.add(int(year_part)); break
            if not available_years:
                 has_price_prefix = any(c.startswith('price_') for c in data.columns)
                 has_vol_prefix = any(c.startswith('available_volume_') for c in data.columns)
                 err_msg = "No columns found matching the 'price_YYYY' or 'available_volume_YYYY' format."
                 if has_price_prefix or has_vol_prefix:
                     err_msg = "Found columns starting with 'price_'/'available_volume_', but couldn't extract valid years (YYYY). Please check column naming convention."
                 return None, err_msg, [], [], []
            for col in list(set(cols_to_convert_numeric)):
                if col in data.columns: data[col] = pd.to_numeric(data[col], errors='coerce')
            data['priority'] = data['priority'].fillna(0)
            for col in data.columns:
                 if col.startswith('available_volume_') and col in year_data_cols_found:
                     data[col] = data[col].fillna(0).apply(lambda x: max(0, int(x)) if pd.notna(x) else 0).clip(lower=0)
                 elif col.startswith('price_') and col in year_data_cols_found:
                     data[col] = data[col].fillna(0.0).apply(lambda x: max(0.0, float(x)) if pd.notna(x) else 0.0).clip(lower=0)
            available_years = sorted(list(available_years))
            invalid_types_found = []
            if 'project_type' in data.columns:
                data['project_type'] = data['project_type'].astype(str).str.lower().str.strip()
                valid_types = ['reduction', 'technical removal', 'natural removal']
                invalid_types_df = data[~data['project_type'].isin(valid_types)]
                if not invalid_types_df.empty:
                     invalid_types_found = invalid_types_df['project_type'].unique().tolist()
                     data = data[data['project_type'].isin(valid_types)].copy()
            else:
                 return None, "Critical error: 'project_type' column missing despite initial check passing.", available_years, [], []
            cols_to_keep = core_cols_std[:]
            for col in optional_cols_std:
                if col in data.columns: cols_to_keep.append(col)
            cols_to_keep.extend(year_data_cols_found)
            data = data[list(set(cols_to_keep))]
            final_rename_map = {
                'project_name': 'project name', 'project_type': 'project type', 'priority': 'priority',
                'description': 'Description', 'project_link': 'Project Link'
            }
            for yr_col in year_data_cols_found:
                final_rename_map[yr_col] = yr_col.replace('_', ' ')
            rename_map_for_df = {k: v for k, v in final_rename_map.items() if k in data.columns}
            data.rename(columns=rename_map_for_df, inplace=True)
            project_names_list = []
            if 'project name' in data.columns:
                 project_names_list = sorted(data['project name'].unique().tolist())
            return data, None, available_years, project_names_list, invalid_types_found

        try:
            data, error_msg, available_years_in_data, project_names_list, invalid_types_found = load_and_prepare_data(df_upload)
            if invalid_types_found:
                 st.sidebar.warning(f"Ignored rows with invalid project types: {', '.join(invalid_types_found)}. Valid types are 'reduction', 'technical removal', 'natural removal'.")
            if error_msg:
                st.sidebar.error(error_msg)
                st.session_state.data_loaded_successfully = False; st.session_state.working_data_full = None
                st.session_state.project_names = []; st.session_state.available_years_in_data = []
                st.session_state.selected_projects = []; st.session_state.annual_targets = {}
            else:
                st.session_state.project_names = project_names_list; st.session_state.available_years_in_data = available_years_in_data
                st.session_state.working_data_full = data; st.session_state.data_loaded_successfully = True
                st.sidebar.success("Data loaded successfully!")
                current_selection = st.session_state.get('selected_projects', [])
                valid_current_selection = [p for p in current_selection if p in project_names_list]
                if not valid_current_selection and project_names_list:
                    st.session_state.selected_projects = project_names_list
                else: st.session_state.selected_projects = valid_current_selection
                st.session_state.annual_targets = {}
        except Exception as e:
            st.sidebar.error(f"An unexpected error occurred during file processing: {e}")
            st.sidebar.error(f"Traceback: {traceback.format_exc()}")
            st.session_state.data_loaded_successfully = False; st.session_state.working_data_full = None
            st.session_state.project_names = []; st.session_state.available_years_in_data = []
            st.session_state.selected_projects = []; st.session_state.annual_targets = {}

    # --- Settings (Displayed only if data loaded successfully) ---
    if st.session_state.get('data_loaded_successfully', False):
        data_for_ui = st.session_state.working_data_full
        available_years_in_data = st.session_state.available_years_in_data
        project_names_list = st.session_state.project_names
        if not available_years_in_data:
            st.sidebar.warning("No usable year data (price/volume columns) found in the uploaded file.")
        else:
            st.markdown("## 2. Portfolio Settings")
            min_year_data = min(available_years_in_data); max_year_data = max(available_years_in_data)
            max_years_slider = min(20, max(1, max_year_data - min_year_data + 1))
            years_to_plan = st.slider(f"Years to Plan (Starting {min_year_data})", 1, max_years_slider, st.session_state.years_slider_sidebar, key='years_slider_sidebar_widget')
            st.session_state.years_slider_sidebar = years_to_plan
            start_year_selected = min_year_data; end_year_selected = start_year_selected + years_to_plan - 1
            selected_years_range = list(range(start_year_selected, end_year_selected + 1))
            actual_years_present_in_data = []
            for year in selected_years_range:
                price_col = f"price {year}"; vol_col = f"available volume {year}"
                if price_col in data_for_ui.columns and vol_col in data_for_ui.columns: actual_years_present_in_data.append(year)
            st.session_state.selected_years = actual_years_present_in_data
            if not st.session_state.selected_years:
                st.sidebar.error(f"No data available for the selected period ({start_year_selected}-{end_year_selected}). Adjust the 'Years to Plan' slider or check your CSV data.")
                st.session_state.actual_start_year = None; st.session_state.actual_end_year = None
            else:
                st.session_state.actual_start_year = min(st.session_state.selected_years); st.session_state.actual_end_year = max(st.session_state.selected_years)
                st.sidebar.markdown(f"Planning Horizon: **{st.session_state.actual_start_year} - {st.session_state.actual_end_year}**")
                st.session_state.constraint_type = st.radio("Constraint Type:", ('Volume', 'Budget'), index=['Volume', 'Budget'].index(st.session_state.get('constraint_type', 'Volume')), key='constraint_type_sidebar_widget', horizontal=True, help="Choose whether annual targets are defined in tons (Volume) or currency (Budget).")
                constraint_type = st.session_state.constraint_type
                st.markdown("### Annual Target Settings")
                master_target_value = st.session_state.get('master_target')
                if constraint_type == 'Volume':
                    default_val_vol = 1000
                    if master_target_value is not None and isinstance(master_target_value, (int, float)):
                        try: default_val_vol = int(float(master_target_value))
                        except (ValueError, TypeError): pass
                    default_target = st.number_input("Default Annual Target Volume (t):", 0, step=100, value=default_val_vol, key='master_volume_sidebar', help="Set a default target volume per year. You can override specific years below.")
                else: # Budget
                    default_val_bud = 100000.0
                    if master_target_value is not None and isinstance(master_target_value, (int, float)):
                        try: default_val_bud = float(master_target_value)
                        except (ValueError, TypeError): pass
                    default_target = st.number_input("Default Annual Target Budget (€):", 0.0, step=1000.0, value=default_val_bud, format="%.2f", key='master_budget_sidebar', help="Set a default target budget per year. You can override specific years below.")
                st.session_state.master_target = default_target
                with st.expander("Customize Annual Targets (+/-)", expanded=False):
                    current_annual_targets = st.session_state.get('annual_targets', {})
                    updated_targets_from_inputs = {}
                    if not st.session_state.selected_years: st.caption("Select years using the 'Years to Plan' slider above first.")
                    else:
                         for year in st.session_state.selected_years:
                             year_target_value = current_annual_targets.get(year, default_target)
                             input_key = f"target_{year}_{constraint_type}"; label = f"Target {year} [t]" if constraint_type == 'Volume' else f"Target {year} [€]"
                             if constraint_type == 'Volume':
                                 try: input_val = int(year_target_value)
                                 except (ValueError, TypeError): input_val = int(default_target)
                                 updated_targets_from_inputs[year] = st.number_input(label, min_value=0, step=100, value=input_val, key=input_key)
                             else:
                                 try: input_val = float(year_target_value)
                                 except (ValueError, TypeError): input_val = float(default_target)
                                 updated_targets_from_inputs[year] = st.number_input(label, min_value=0.0, step=1000.0, value=input_val, format="%.2f", key=input_key)
                    st.session_state.annual_targets = updated_targets_from_inputs
                st.sidebar.markdown("### Allocation Goal & Preferences")
                min_fulfill = st.sidebar.slider(f"Min. Target Fulfillment (%)", 50, 100, st.session_state.get('min_fulfillment_perc', 95), help=f"The allocation algorithm will attempt to allocate at least this percentage of the annual target {constraint_type} by potentially adding more volume/budget in an adjustment step.", key='min_fulfill_perc_sidebar')
                st.session_state.min_fulfillment_perc = min_fulfill
                min_chunk_val = st.sidebar.number_input("Min. Allocation Unit (t)", 1, step=1, value=st.session_state.get('min_alloc_chunk', 1), help="Smallest amount (in tons) to allocate from any single project in a year. Prevents trivial allocations.", key='min_alloc_chunk_sidebar')
                st.session_state.min_alloc_chunk = int(min_chunk_val)
                st.sidebar.markdown("### Removal Volume Transition (If 'Reduction' Projects Used)")
                reduction_present = 'reduction' in data_for_ui['project type'].unique()
                if reduction_present: st.sidebar.info("Transition settings apply because 'Reduction' type projects are present in your data and might be selected.")
                else: st.sidebar.info("Note: Transition settings are currently inactive as no 'Reduction' type projects were found in the loaded data.")
                removal_help = f"Target percentage of total allocated volume coming from Removal projects (Technical + Natural) in the final year ({st.session_state.actual_end_year}). This guides the portfolio mix over time *only if* 'Reduction' projects are selected and used in the portfolio."
                rem_target_slider = st.sidebar.slider(f"Target Removal Vol % ({st.session_state.actual_end_year})", 0, 100, int(st.session_state.get('removal_target_end_year', 0.8)*100), help=removal_help, key='removal_perc_slider_sidebar', disabled=not reduction_present)
                st.session_state.removal_target_end_year = rem_target_slider / 100.0
                st.session_state.transition_speed = st.sidebar.slider("Transition Speed", 1, 10, st.session_state.get('transition_speed', 5), help="Speed of ramping up to the target removal percentage (1=Slowest, 10=Fastest). Active only if 'Reduction' projects are used.", key='transition_speed_slider_sidebar', disabled=not reduction_present)
                st.sidebar.markdown("### Removal Category Preference")
                removal_types_present = any(pt in data_for_ui['project type'].unique() for pt in ['technical removal', 'natural removal'])
                rem_pref_val = st.sidebar.slider("Technical vs Natural Preference", 1, 10, st.session_state.get('removal_preference_slider', 5), format="%d", key='removal_pref_slider_sidebar', help="Adjusts the target split between Technical and Natural Removals. 1 leans strongly towards Natural, 5 is balanced, 10 leans strongly towards Technical.", disabled=not removal_types_present)
                st.session_state['removal_preference_slider'] = rem_pref_val
                tech_pref_ratio = (rem_pref_val - 1) / 9.0
                st.session_state.category_split = {'technical removal': tech_pref_ratio, 'natural removal': 1.0 - tech_pref_ratio}
                st.sidebar.markdown("## 3. Select Projects")
                if not project_names_list: st.sidebar.warning("No projects available from the loaded data.")
                else:
                    st.session_state.selected_projects = st.sidebar.multiselect("Select projects to include:", options=project_names_list, default=st.session_state.get('selected_projects', project_names_list), key='project_selector_sidebar')
                    if 'priority' in data_for_ui.columns:
                        boost_options = [p for p in project_names_list if p in st.session_state.selected_projects]
                        if boost_options:
                            current_favorite = st.session_state.get('favorite_projects_selection', [])
                            valid_default_favorite = [f for f in current_favorite if f in boost_options][:1]
                            st.session_state.favorite_projects_selection = st.sidebar.multiselect("Favorite Project (Priority Boost):", options=boost_options, default=valid_default_favorite, key='favorite_selector_sidebar', max_selections=1, help="Optionally select one project to receive a small priority boost during allocation.")
                        else: st.sidebar.info("Select projects above to enable the Favorite Project boost option."); st.session_state.favorite_projects_selection = []
                    else: st.sidebar.info("Favorite Project boost disabled: No 'priority' column found in the loaded data."); st.session_state.favorite_projects_selection = []


# ==================================
# Main Page Content
# ==================================
# Use markdown for a styled title - REMOVED text-align: center;
st.markdown(f"<h1 style='color: #8ca734;'>Carbon Portfolio Builder</h1>", unsafe_allow_html=True)
st.markdown("---") # Add a visual separator

# --- Prerequisite Checks ---
if not st.session_state.get('data_loaded_successfully', False):
    st.info("👋 Welcome! Please upload your project data CSV using the sidebar menu to get started.")

# --- Project Details Expander (Show only if data is loaded) ---
elif st.session_state.get('data_loaded_successfully', False):
    # ADDED: Title and caption before the expander
    st.markdown("## Project Offerings")
    st.caption("To get more information on the specific project please unfold the list and click the link to the project slide.")

    with st.expander("View Project Details"):
        project_df = st.session_state.working_data_full
        if project_df is not None and not project_df.empty:
            display_cols_options = ['project name', 'project type']
            column_config = {}
            if 'Description' in project_df.columns:
                display_cols_options.append('Description')
            if 'Project Link' in project_df.columns:
                display_cols_options.append('Project Link')
                column_config["Project Link"] = st.column_config.LinkColumn(
                         "Project Link", display_text="Visit ->", help="Click link to visit project page (if available)"
                     )
            st.dataframe(
                 project_df[display_cols_options],
                 column_config=column_config,
                 hide_index=True,
                 use_container_width=True
                 )
        else:
             st.write("Project data is loaded but appears to be empty.")
    st.markdown("---") # Separator after the expander


# --- Main Calculation/Display Logic ---
if st.session_state.get('data_loaded_successfully', False):
    if not st.session_state.get('selected_projects'):
        st.warning("⚠️ Please select the projects you want to include in the portfolio using the sidebar (Section 3).")
    elif not st.session_state.get('selected_years'):
         st.warning("⚠️ No valid years identified for the selected planning horizon. Please adjust the 'Years to Plan' slider (Section 2) or verify the year columns in your CSV data.")
    else:
        required_keys = [
            'working_data_full', 'selected_projects', 'selected_years',
            'actual_start_year', 'actual_end_year', 'constraint_type',
            'annual_targets', 'removal_target_end_year', 'transition_speed',
            'category_split', 'favorite_projects_selection', 'min_fulfillment_perc',
            'min_alloc_chunk'
        ]
        keys_present = all(k in st.session_state and st.session_state.get(k) is not None for k in required_keys if k not in ['annual_targets', 'favorite_projects_selection'])
        keys_present = keys_present and ('annual_targets' in st.session_state) and ('favorite_projects_selection' in st.session_state)

        if keys_present:
            try:
                # --- Get Settings from Session State for Allocation ---
                fav_proj = st.session_state.favorite_projects_selection[0] if st.session_state.favorite_projects_selection else None
                constraint = st.session_state.constraint_type
                min_chunk = st.session_state.min_alloc_chunk
                annual_targets_to_use = st.session_state.annual_targets

                # --- Display Info Messages based on Settings ---
                if constraint == 'Budget':
                    st.info(f"**Budget Mode:** Projects initially receive budget based on weighted priority and price. A project might receive 0 volume if its allocated budget share is less than the cost of {min_chunk}t. The adjustment step may allocate more later if needed to meet the fulfillment goal.")
                st.success(f"**Allocation Goal:** The calculation will attempt to fulfill at least **{st.session_state.min_fulfillment_perc}%** of the specified annual target {constraint} for each year.")

                # --- Run Allocation Function ---
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

                # --- Prepare Detailed Dataframe from Results ---
                details_list = []
                if results:
                    for year_v, projects_v in results.items():
                        if projects_v:
                            for proj_v in projects_v:
                                if isinstance(proj_v, dict) and (proj_v.get('allocated_volume', 0) >= min_chunk or proj_v.get('allocated_cost', 0) > 1e-6):
                                    details_list.append({'year': year_v, 'project name': proj_v.get('project name'), 'type': proj_v.get('type'), 'volume': proj_v.get('allocated_volume', 0), 'price': proj_v.get('price_used', None), 'cost': proj_v.get('allocated_cost', 0.0)})
                details_df = pd.DataFrame(details_list)
                pivot_display = pd.DataFrame()

                # --- Display Metrics & Pie Chart Layout ---
                st.markdown("## Portfolio Summary")
                col_l, col_r = st.columns([2, 1.2], gap="large")
                with col_l:
                    st.markdown("#### Key Metrics (Overall)")
                    if not summary.empty:
                         total_cost_all_years = summary['Allocated Cost'].sum(); total_volume_all_years = summary['Allocated Volume'].sum()
                         overall_avg_price = total_cost_all_years / total_volume_all_years if total_volume_all_years > 0 else 0.0
                         st.markdown(f"""<div class="metric-box"><b>Total Portfolio Cost</b> € {total_cost_all_years:,.2f}</div>""", unsafe_allow_html=True)
                         st.markdown(f"""<div class="metric-box"><b>Total Portfolio Volume</b> {total_volume_all_years:,.0f} t</div>""", unsafe_allow_html=True)
                         st.markdown(f"""<div class="metric-box"><b>Overall Average Price</b> € {overall_avg_price:,.2f} /t</div>""", unsafe_allow_html=True)
                    else:
                         st.markdown(f"""<div class="metric-box"><b>Total Portfolio Cost</b> - </div>""", unsafe_allow_html=True)
                         st.markdown(f"""<div class="metric-box"><b>Total Portfolio Volume</b> - </div>""", unsafe_allow_html=True)
                         st.markdown(f"""<div class="metric-box"><b>Overall Average Price</b> - </div>""", unsafe_allow_html=True)
                         st.warning("Could not calculate overall portfolio metrics. Allocation might be empty.")
                with col_r:
                    st.markdown("#### Volume by Project Type")
                    if not details_df.empty:
                        pie_data = details_df.groupby('type')['volume'].sum().reset_index()
                        pie_data = pie_data[pie_data['volume'] > 1e-6]
                        if not pie_data.empty:
                            st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
                            fig_pie = px.pie(pie_data, values='volume', names='type', color='type', color_discrete_map=type_color_map)
                            fig_pie.update_layout(showlegend=True, legend_title_text='Project Type', legend_orientation="h", legend_yanchor="bottom", legend_y=-0.2, legend_xanchor="center", legend_x=0.5, margin=dict(t=5, b=50, l=0, r=0), height=350)
                            fig_pie.update_traces(textposition='inside', textinfo='percent', sort=False, hole=.3, marker=dict(line=dict(color='#FFFFFF', width=1)))
                            st.plotly_chart(fig_pie, use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        elif not details_df.empty and pie_data.empty: st.caption("No significant volume allocated to display in the pie chart.")
                        else: st.caption("Could not generate pie chart data from allocation details.")
                    else: st.caption("No allocation details available to generate the pie chart.")
                st.markdown("---")

                # --- Visualization Section (Composition Plot) ---
                if details_df.empty and summary.empty: st.warning("No allocation data generated to display plots or tables.")
                elif details_df.empty: st.warning("No detailed project allocations available to create the composition plot.")
                else:
                    st.markdown("### Portfolio Composition & Price Over Time")
                    summary_plot_data = details_df.groupby(['year', 'type']).agg(volume=('volume', 'sum'), cost=('cost', 'sum')).reset_index()
                    price_summary_data = pd.DataFrame()
                    if not summary.empty: price_summary_data = summary[['Year', 'Avg. Price']].rename(columns={'Year':'year', 'Avg. Price':'avg_price'})
                    fig_composition = make_subplots(specs=[[{"secondary_y": True}]])
                    y_metric = 'volume' if constraint == 'Volume' else 'cost'; y_label = 'Allocated Volume (t)' if constraint == 'Volume' else 'Allocated Cost (€)'
                    y_format = '{:,.0f}' if constraint == 'Volume' else '€{:,.2f}'; y_hover_label = 'Volume' if constraint == 'Volume' else 'Cost'
                    type_order = ['reduction', 'natural removal', 'technical removal']; types_in_results = details_df['type'].unique()
                    for t_name in type_order:
                        if t_name in types_in_results:
                            df_type = summary_plot_data[summary_plot_data['type'] == t_name]
                            if not df_type.empty and y_metric in df_type.columns and df_type[y_metric].sum() > 1e-6:
                                fig_composition.add_trace(go.Bar(x=df_type['year'], y=df_type[y_metric], name=t_name.replace('_', ' ').capitalize(), marker_color=type_color_map.get(t_name, default_color), hovertemplate=f'Year: %{{x}}<br>Type: {t_name.replace("_", " ").capitalize()}<br>{y_hover_label}: %{{y:{y_format}}}<extra></extra>'), secondary_y=False)
                    if not price_summary_data.empty: fig_composition.add_trace(go.Scatter(x=price_summary_data['year'], y=price_summary_data['avg_price'], name='Avg Price (€/t)', mode='lines+markers', marker=dict(symbol='circle', size=8), line=dict(color='#1B5E20', width=3), hovertemplate='Year: %{x}<br>Avg Price: €%{y:,.2f}/t<extra></extra>'), secondary_y=True)
                    if not summary.empty and 'Actual Removal Vol %' in summary.columns: fig_composition.add_trace(go.Scatter(x=summary['Year'], y=summary['Actual Removal Vol %'], name='Actual Removal Vol %', mode='lines+markers', line=dict(color='darkorange', dash='dash'), marker=dict(symbol='star', size=8), hovertemplate='Year: %{x}<br>Actual Removal: %{y:.1f}%<extra></extra>'), secondary_y=True)
                    y2_max_limit = 105
                    if not price_summary_data.empty and not price_summary_data['avg_price'].empty: y2_max_limit = max(y2_max_limit, price_summary_data['avg_price'].max() * 1.1)
                    if not summary.empty and 'Actual Removal Vol %' in summary.columns and not summary['Actual Removal Vol %'].empty: y2_max_limit = max(y2_max_limit, summary['Actual Removal Vol %'].max() * 1.1)
                    fig_composition.update_layout(xaxis_title='Year', yaxis_title=y_label, yaxis2_title='Avg Price (€/t) / Actual Removal %', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), barmode='stack', template="plotly_white", margin=dict(t=20, l=0, r=0, b=0), yaxis=dict(rangemode='tozero'), yaxis2=dict(rangemode='tozero', range=[0, y2_max_limit]), hovermode="x unified")
                    if st.session_state.selected_years: fig_composition.update_xaxes(tickmode='array', tickvals=st.session_state.selected_years, dtick=1)
                    st.plotly_chart(fig_composition, use_container_width=True)

                # --- Display Combined Detailed Allocation Table ---
                st.markdown("### Detailed Allocation by Project and Year")
                if not details_df.empty or not summary.empty:
                    try:
                        pivot_final = pd.DataFrame()
                        years_present_in_results = st.session_state.selected_years
                        if not details_df.empty:
                            pivot_intermediate = pd.pivot_table(details_df, values=['volume', 'cost', 'price'], index=['project name', 'type'], columns='year', aggfunc={'volume': 'sum', 'cost': 'sum', 'price': 'first'})
                            if not pivot_intermediate.empty:
                                pivot_final = pivot_intermediate.swaplevel(0, 1, axis=1)
                                metric_order = ['volume', 'cost', 'price']
                                years_present_in_pivot = sorted(pivot_final.columns.get_level_values(0).unique())
                                years_present_in_results = years_present_in_pivot
                                final_multi_index = pd.MultiIndex.from_product([years_present_in_results, metric_order], names=['year', 'metric'])
                                pivot_final = pivot_final.reindex(columns=final_multi_index)
                                pivot_final = pivot_final.sort_index(axis=1, level=[0, 1])
                                pivot_final.index.names = ['Project Name', 'Type']
                        total_data = {}
                        if not summary.empty:
                             summary_indexed = summary.set_index('Year')
                             for year in years_present_in_results:
                                 if year in summary_indexed.index: vol, cost, avg_price = summary_indexed.loc[year, ['Allocated Volume', 'Allocated Cost', 'Avg. Price']]
                                 else: vol, cost, avg_price = 0, 0.0, 0.0
                                 total_data[(year, 'volume')] = vol; total_data[(year, 'cost')] = cost; total_data[(year, 'price')] = avg_price
                        total_row_index = pd.MultiIndex.from_tuples([('Total Portfolio', 'All Types')], names=['Project Name', 'Type'])
                        total_row_df = pd.DataFrame(total_data, index=total_row_index)
                        if pivot_final.empty: pivot_display = total_row_df
                        else: pivot_display = pd.concat([pivot_final, total_row_df])
                        pivot_display = pivot_display.fillna(0)
                        formatter = {}
                        for year, metric in pivot_display.columns:
                            if metric == 'volume': formatter[(year, metric)] = '{:,.0f} t'
                            elif metric == 'cost': formatter[(year, metric)] = '€{:,.2f}'
                            elif metric == 'price': formatter[(year, metric)] = lambda x: f'€{x:,.2f}/t' if pd.notna(x) and x != 0 else ('-' if metric == 'price' else '€0.00/t')
                        st.dataframe(pivot_display.style.format(formatter, na_rep="-"), use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not create or display the detailed allocation table: {e}")
                        st.error(f"Traceback: {traceback.format_exc()}")
                else: st.info("No allocation details or summary data available to display.")

                # --- Download Button ---
                if not pivot_display.empty:
                    csv_df = pivot_display.copy()
                    csv_df.columns = [f"{int(col[0])}_{col[1]}" for col in csv_df.columns.values]
                    csv_df = csv_df.reset_index()
                    csv_string = csv_df.to_csv(index=False).encode('utf-8')
                    st.markdown("---") # Separator before download button

                    # REMOVED columns for centering - button will now appear on the left
                    st.download_button(
                       label="Download Detailed Allocation (CSV)",
                       data=csv_string,
                       file_name=f"portfolio_allocation_{datetime.date.today()}.csv",
                       mime='text/csv',
                       key='download-csv'
                       # Styling is applied via the global CSS block at the top
                    )
                # --- End Download Button ---

            # --- Error Handling for Allocation/Display ---
            except ValueError as e: st.error(f"Configuration or Allocation Error: {e}")
            except KeyError as e: st.error(f"Data Error: Missing expected data key: '{e}'. Please check your CSV format (column names like 'project name', 'price X', etc.) and your selections.")
            except Exception as e:
                st.error(f"An unexpected error occurred during portfolio calculation or display: {e}")
                st.error(f"Traceback: {traceback.format_exc()}")
        # --- End Main Logic ('if keys_present:') ---
        else:
            st.error("Critical Error: Required settings are missing from the application state. This might happen after code changes or unexpected resets. Please try reloading the data and re-configuring the settings in the sidebar.")

# --- Footer ---
st.markdown("---")
try:
    zurich_tz = pytz.timezone('Europe/Zurich')
    now_zurich = datetime.datetime.now(zurich_tz)
    st.caption(f"Report generated: {now_zurich.strftime('%Y-%m-%d %H:%M:%S %Z')}")
except Exception as e:
    st.error(f"Timezone processing error: {e}")
    now_local = datetime.datetime.now()
    st.caption(f"Report generated: {now_local.strftime('%Y-%m-%d %H:%M:%S')} (Timezone: Server Local)")
