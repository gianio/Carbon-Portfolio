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
# (Allocation function remains largely unchanged in its core logic)
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
    (Docstring details omitted for brevity)
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
    # Ensure required columns for allocation are present (margin columns are handled later)
    required_base_cols = ['project name', 'project type', 'priority']
    price_cols_needed, volume_cols_needed = [], []
    for year in selected_years:
        price_cols_needed.append(f"price {year}")
        volume_cols_needed.append(f"available volume {year}")

    missing_base = [col for col in required_base_cols if col not in project_data_selected.columns]
    if missing_base:
        raise ValueError(f"Input data is missing required base columns for allocation: {', '.join(missing_base)}")

    missing_years_data = []
    for year in selected_years:
        if f"price {year}" not in project_data_selected.columns:
            missing_years_data.append(f"price {year}")
        if f"available volume {year}" not in project_data_selected.columns:
            missing_years_data.append(f"available volume {year}")

    if missing_years_data:
        years_affected = sorted(list(set(int(col.split()[-1]) for col in missing_years_data if col.split()[-1].isdigit())))
        raise ValueError(f"Input data is missing price/volume information for required year(s) for allocation: {', '.join(map(str, years_affected))}.")

    # Convert relevant columns to numeric, handle NaNs and negatives (for allocation columns)
    numeric_cols_to_check = ['priority'] + price_cols_needed + volume_cols_needed
    for col in numeric_cols_to_check:
        if col in project_data_selected.columns: # Should be present due to checks above
            project_data_selected[col] = pd.to_numeric(project_data_selected[col], errors='coerce') # Coerce errors for safety
            if col == 'priority':
                project_data_selected[col] = project_data_selected[col].fillna(0)
            elif col.startswith("available volume"):
                project_data_selected[col] = project_data_selected[col].fillna(0).apply(lambda x: int(x) if pd.notna(x) and x >= 0 else 0).clip(lower=0)
            elif col.startswith("price"):
                 project_data_selected[col] = project_data_selected[col].fillna(0.0).apply(lambda x: float(x) if pd.notna(x) and x >= 0 else 0.0).clip(lower=0.0)


    all_project_types_in_selection = project_data_selected['project type'].unique()
    is_reduction_selected = 'reduction' in all_project_types_in_selection
    total_years_duration = end_year_portfolio - start_year_portfolio

    # --- Yearly Allocation Loop ---
    for year in selected_years:
        yearly_target = annual_targets.get(year, 0)
        price_col = f"price {year}"
        volume_col = f"available volume {year}"

        year_total_allocated_vol = 0
        year_total_allocated_cost = 0.0
        summary_template = {
            'Year': year, f'Target {constraint_type}': yearly_target, 'Allocated Volume': 0,
            'Allocated Cost': 0.0, 'Avg. Price': 0.0, 'Actual Removal Vol %': 0.0,
            'Target Removal Vol %': 0.0, 'Total Yearly Margin': 0.0 # New field for summary
        }

        if yearly_target <= 0:
            yearly_summary_list.append(summary_template); portfolio_details[year] = []; continue

        target_percentages = {}
        if is_reduction_selected:
            start_removal_percent = 0.10; end_removal_percent = removal_target_percent_end_year
            progress = 1.0 if total_years_duration <= 0 else max(0, min(1, (year - start_year_portfolio) / total_years_duration))
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
            if total_alloc_share > 1e-9: # Normalize if only one type is selected but preferences might exist for both
                target_percentages['technical removal'] = (tech_alloc_share / total_alloc_share) if tech_selected else 0.0
                target_percentages['natural removal'] = (nat_alloc_share / total_alloc_share) if nat_selected else 0.0
            elif tech_selected or nat_selected: # If only one selected and no preference, it gets 100%
                 target_percentages['technical removal'] = 1.0 if tech_selected and not nat_selected else (0.5 if tech_selected and nat_selected else 0.0)
                 target_percentages['natural removal'] = 1.0 if nat_selected and not tech_selected else (0.5 if tech_selected and nat_selected else 0.0)
            else: # No removal types selected
                 target_percentages['technical removal'] = 0.0
                 target_percentages['natural removal'] = 0.0
            target_percentages['reduction'] = 0.0


        current_sum = sum(target_percentages.values())
        if abs(current_sum - 1.0) > 1e-6 and current_sum > 0: # Normalize if sum is not 1
            norm_factor = 1.0 / current_sum
            target_percentages = {ptype: share * norm_factor for ptype, share in target_percentages.items()}
        summary_template['Target Removal Vol %'] = (target_percentages.get('technical removal', 0) + target_percentages.get('natural removal', 0)) * 100

        projects_year_df = project_data_selected[(project_data_selected[price_col] > 0) & (project_data_selected[volume_col] >= min_allocation_chunk)].copy()
        projects_year_df['initial_allocated_volume'] = 0; projects_year_df['initial_allocated_cost'] = 0.0
        projects_year_df['final_priority'] = np.nan

        if projects_year_df.empty:
            yearly_summary_list.append(summary_template); portfolio_details[year] = []; continue

        for project_type_alloc in all_project_types_in_selection: # Renamed project_type
            target_share = target_percentages.get(project_type_alloc, 0)
            if target_share <= 0: continue
            target_resource = yearly_target * target_share
            projects_of_type = projects_year_df[projects_year_df['project type'] == project_type_alloc].copy()
            if projects_of_type.empty: continue

            total_priority_in_type = projects_of_type['priority'].sum()
            if total_priority_in_type <= 0: # Equal weight if no priorities or all zero
                num_projects_in_type = len(projects_of_type)
                projects_of_type['norm_prio_base'] = (1.0 / num_projects_in_type) if num_projects_in_type > 0 else 0
            else:
                projects_of_type['norm_prio_base'] = projects_of_type['priority'] / total_priority_in_type

            current_priorities = projects_of_type.set_index('project name')['norm_prio_base'].to_dict()
            final_priorities = current_priorities.copy()

            if favorite_project and favorite_project in final_priorities and projects_of_type.loc[projects_of_type['project name'] == favorite_project, 'project type'].iloc[0] == project_type_alloc :
                fav_proj_base_prio = current_priorities[favorite_project]; boost_factor = priority_boost_percent / 100.0
                priority_increase = fav_proj_base_prio * boost_factor; new_fav_proj_prio = fav_proj_base_prio + priority_increase
                other_projects = [p for p in current_priorities if p != favorite_project]
                sum_other_priorities = sum(current_priorities[p] for p in other_projects)
                temp_priorities = {favorite_project: new_fav_proj_prio}; reduction_factor = 0
                if sum_other_priorities > 1e-9: reduction_factor = priority_increase / sum_other_priorities # Distribute the boosted amount proportionally from others
                for name in other_projects: temp_priorities[name] = max(0, current_priorities[name] * (1 - reduction_factor)) # Ensure prio doesn't go negative
                total_final_prio = sum(temp_priorities.values())
                if total_final_prio > 1e-9: final_priorities = {p: prio / total_final_prio for p, prio in temp_priorities.items()} # Re-normalize
                elif favorite_project in temp_priorities : final_priorities = {favorite_project: 1.0} # Edge case if only fav project remains

            project_weights = {}; total_weight = 0
            if constraint_type == 'Budget':
                for _, row_budget in projects_of_type.iterrows(): # Renamed row
                    name = row_budget['project name']; final_prio = final_priorities.get(name, 0); price = row_budget[price_col]
                    weight = final_prio * price if price > 0 else 0; project_weights[name] = weight; total_weight += weight

            for idx, row_alloc in projects_of_type.iterrows(): # Renamed row
                name = row_alloc['project name']; final_prio = final_priorities.get(name, 0); available_vol = row_alloc[volume_col]; price = row_alloc[price_col]
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
                        target_volume_proj = target_budget_proj / price if price > 0 else 0
                        allocated_volume = min(target_volume_proj, available_vol)
                    else: # No weights, perhaps only one project or all prices zero
                        if len(projects_of_type) == 1: # Allocate full budget resource if only one project
                             target_volume_proj = target_resource / price if price > 0 else 0
                             allocated_volume = min(target_volume_proj, available_vol)
                        else: # Multiple projects, no prices, can't allocate budget fairly without volume target
                            allocated_volume = 0


                allocated_volume = int(max(0, math.floor(allocated_volume / min_allocation_chunk) * min_allocation_chunk)) # Allocate in chunks

                if allocated_volume >= min_allocation_chunk:
                    allocated_cost = allocated_volume * price
                    projects_year_df.loc[projects_year_df['project name'] == name, 'initial_allocated_volume'] += allocated_volume
                    projects_year_df.loc[projects_year_df['project name'] == name, 'initial_allocated_cost'] += allocated_cost
                    year_total_allocated_vol += allocated_volume; year_total_allocated_cost += allocated_cost

        target_threshold = yearly_target * min_target_fulfillment_percent
        current_metric_total = year_total_allocated_cost if constraint_type == 'Budget' else year_total_allocated_vol

        if current_metric_total < target_threshold and yearly_target > 0 :
            needed = target_threshold - current_metric_total
            projects_year_df['remaining_volume'] = projects_year_df[volume_col] - projects_year_df['initial_allocated_volume']
            # Sort by priority (higher is better), then by price (lower is better for adjustment)
            adjustment_candidates = projects_year_df[
                (projects_year_df['remaining_volume'] >= min_allocation_chunk) &
                (projects_year_df[price_col] > 0)
            ].sort_values(by=['priority', price_col], ascending=[False, True]).copy()


            for idx_adj, row_adj in adjustment_candidates.iterrows(): # Renamed idx, row
                if needed <= (1e-2 if constraint_type=='Budget' else 0): break # Small tolerance for budget
                name_adj = row_adj['project name']; price_adj = row_adj[price_col]; available_for_adj = row_adj['remaining_volume']
                volume_to_add = 0; cost_to_add = 0.0

                if constraint_type == 'Volume':
                    add_vol = min(available_for_adj, needed)
                else: # Budget
                    max_affordable_vol = needed / price_adj if price_adj > 0 else 0
                    add_vol = min(available_for_adj, max_affordable_vol)

                add_vol_chunked = int(math.floor(add_vol / min_allocation_chunk) * min_allocation_chunk)

                if add_vol_chunked >= min_allocation_chunk:
                    cost_increase = add_vol_chunked * price_adj
                    # Check if adding this chunk makes sense
                    # For budget, ensure we don't drastically overshoot 'needed' unless it's the only chunk size.
                    if constraint_type == 'Volume' or (cost_increase <= needed * 1.1 or cost_increase < price_adj * min_allocation_chunk * 1.5):
                        volume_to_add = add_vol_chunked
                        cost_to_add = cost_increase
                        needed -= cost_to_add if constraint_type == 'Budget' else volume_to_add

                if volume_to_add > 0:
                    # Use .loc with the original index from projects_year_df
                    original_df_idx = projects_year_df[projects_year_df['project name'] == name_adj].index[0]
                    projects_year_df.loc[original_df_idx, 'initial_allocated_volume'] += volume_to_add
                    projects_year_df.loc[original_df_idx, 'initial_allocated_cost'] += cost_to_add
                    # No need to update 'remaining_volume' here as it's re-read if loop continued (but loop breaks often)
                    year_total_allocated_vol += volume_to_add; year_total_allocated_cost += cost_to_add


        final_allocations_list = []
        final_year_allocations_df = projects_year_df[projects_year_df['initial_allocated_volume'] >= min_allocation_chunk].copy()
        for _, row_final in final_year_allocations_df.iterrows(): # Renamed idx, row
            current_price = row_final.get(price_col, None)
            final_allocations_list.append({
                'project name': row_final['project name'],
                'type': row_final['project type'],
                'allocated_volume': row_final['initial_allocated_volume'],
                'allocated_cost': row_final['initial_allocated_cost'],
                'price_used': current_price, # This is the "price YEAR" for margin calculation
                'priority_applied': row_final['final_priority']
            })
        portfolio_details[year] = final_allocations_list
        summary_template['Allocated Volume'] = year_total_allocated_vol
        summary_template['Allocated Cost'] = year_total_allocated_cost
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
# Margin Calculation Functions
# ==================================
def get_margin_per_unit(project_data_row: pd.Series, allocated_price_this_year: float) -> float:
    """
    Calculates the margin per unit for a project based on its pricing model and the allocated price.
    """
    if pd.isna(allocated_price_this_year) or allocated_price_this_year < 0: # Negative price makes no sense for margin from sale
        return 0.0

    # Standardized column names (already renamed with spaces in project_data_row)
    base_price_col = 'base price'
    threshold_price_col = 'threshold price'
    margin_share_col = 'margin share' # Assumed decimal, e.g., 0.1 for 10%
    fixed_purchase_price_col = 'fixed purchase price'
    percental_margin_share_col = 'percental margin share' # Assumed decimal

    margin_per_unit = 0.0

    # Method 1: Base Price, Threshold Price, Margin Share
    bp = project_data_row.get(base_price_col)
    tp = project_data_row.get(threshold_price_col)
    ms = project_data_row.get(margin_share_col)

    if pd.notna(bp) and pd.notna(tp) and pd.notna(ms):
        # Ensure they are numeric, though load_and_prepare should handle this
        try:
            bp = float(bp)
            tp = float(tp)
            ms = float(ms)
            # Margin from share is only on price above threshold
            margin_from_share = max(0, allocated_price_this_year - tp) * ms
            margin_from_base_to_threshold = tp - bp
            margin_per_unit = margin_from_share + margin_from_base_to_threshold
        except (ValueError, TypeError):
            margin_per_unit = 0.0 # Fallback if conversion fails here
            # This path should ideally not be hit if data prep is robust

    # Method 2: Fixed Purchase Price
    elif pd.notna(project_data_row.get(fixed_purchase_price_col)):
        try:
            fpp = float(project_data_row.get(fixed_purchase_price_col))
            margin_per_unit = allocated_price_this_year - fpp
        except (ValueError, TypeError):
             margin_per_unit = 0.0

    # Method 3: Percental Margin Share
    elif pd.notna(project_data_row.get(percental_margin_share_col)):
        try:
            pms = float(project_data_row.get(percental_margin_share_col))
            margin_per_unit = allocated_price_this_year * pms # Assumes pms is decimal 0.xx
        except (ValueError, TypeError):
            margin_per_unit = 0.0
    
    return margin_per_unit if pd.notna(margin_per_unit) and margin_per_unit > -float('inf') else 0.0


def add_margins_to_details_df(details_df: pd.DataFrame, project_master_data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'margin_per_unit' and 'margin' (total yearly project margin) to the details DataFrame.
    """
    if details_df.empty or project_master_data.empty:
        details_df['margin_per_unit'] = 0.0
        details_df['margin'] = 0.0
        return details_df

    details_with_margins = []
    
    # Prepare project_master_data for easy lookup
    project_lookup = project_master_data.set_index('project name')

    for _, row in details_df.iterrows():
        project_name = row['project name']
        allocated_volume = row['volume']
        price_used = row['price'] # This is the 'price_used' from allocation

        new_row = row.to_dict()
        margin_val = 0.0
        margin_pu_val = 0.0

        if project_name in project_lookup.index:
            project_data_row = project_lookup.loc[project_name]
            if pd.notna(price_used) and allocated_volume > 0:
                margin_pu_val = get_margin_per_unit(project_data_row, price_used)
                margin_val = margin_pu_val * allocated_volume
        
        new_row['margin_per_unit'] = margin_pu_val
        new_row['margin'] = margin_val
        details_with_margins.append(new_row)

    return pd.DataFrame(details_with_margins)


# ==================================
# Streamlit App Layout & Logic
# ==================================

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 1. Load Data")
    df_upload = st.file_uploader(
        "Upload Project Data CSV", type="csv", key="uploader_sidebar",
        help="CSV required columns: `project name`, `project type`, `priority`. Needs `price_YYYY` & `available_volume_YYYY`. Optional margin columns: `base price`, `threshold price`, `margin share`, `fixed purchase price`, `percental margin share`. Optional: `description`, `project_link`."
    )
    default_values = {'working_data_full': None, 'selected_years': [], 'selected_projects': [], 'project_names': [], 'favorite_projects_selection': [], 'actual_start_year': None, 'actual_end_year': None, 'available_years_in_data': [], 'constraint_type': 'Volume', 'removal_target_end_year': 0.8, 'transition_speed': 5, 'category_split': {'technical removal': 0.5, 'natural removal': 0.5}, 'annual_targets': {}, 'master_target': None, 'data_loaded_successfully': False, 'years_slider_sidebar': 5, 'min_fulfillment_perc': 95, 'removal_preference_slider': 5, 'min_alloc_chunk': 1}
    for key, default_value in default_values.items():
        if key not in st.session_state: st.session_state[key] = default_value

    if df_upload:
        @st.cache_data
        def load_and_prepare_data(uploaded_file):
            try:
                data = pd.read_csv(uploaded_file)
                data.columns = data.columns.str.lower().str.strip().str.replace(' ', '_')
            except Exception as read_error:
                return None, f"Error reading CSV file: {read_error}", [], [], []

            core_cols_std = ['project_name', 'project_type', 'priority']
            optional_cols_std = ['description', 'project_link']
            # NEW: Define standard names for margin columns
            margin_cols_std = ['base_price', 'threshold_price', 'margin_share', 'fixed_purchase_price', 'percental_margin_share']

            missing_essential = [col for col in core_cols_std if col not in data.columns]
            if missing_essential:
                return None, f"CSV is missing essential columns: {', '.join(missing_essential)}", [], [], []

            # Ensure all potential margin columns exist, fill with NaN if not in original CSV
            for m_col in margin_cols_std:
                if m_col not in data.columns:
                    data[m_col] = np.nan

            numeric_prefixes_std = ['price_', 'available_volume_']
            cols_to_convert_numeric = ['priority'] + margin_cols_std # Add margin cols here
            available_years = set()
            year_data_cols_found = []

            for col in data.columns:
                for prefix in numeric_prefixes_std:
                    year_part = col[len(prefix):]
                    if col.startswith(prefix) and year_part.isdigit():
                        cols_to_convert_numeric.append(col)
                        year_data_cols_found.append(col)
                        available_years.add(int(year_part))
                        break
            
            if not available_years: # Check if any year-specific price/volume data was found
                has_price_prefix = any(c.startswith('price_') for c in data.columns)
                has_vol_prefix = any(c.startswith('available_volume_') for c in data.columns)
                err_msg = "No columns found matching the 'price_YYYY' or 'available_volume_YYYY' format, which are required for allocation."
                if has_price_prefix or has_vol_prefix : err_msg = "Found columns starting with 'price_'/'available_volume_', but couldn't extract valid years (YYYY). Please check column naming convention."
                return None, err_msg, [], [], []


            for col in list(set(cols_to_convert_numeric)): # Use set to avoid duplicates
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce') # Coerce errors to NaN

            # Handle NaNs/negatives specifically for core & margin columns after numeric conversion
            data['priority'] = data['priority'].fillna(0).clip(lower=0)
            for m_col in margin_cols_std: # Margin columns can be NaN if not applicable, prices should be non-negative
                 if m_col in data.columns: # Should be, as added above
                    if m_col in ['base_price', 'threshold_price', 'fixed_purchase_price']:
                         data[m_col] = data[m_col].apply(lambda x: x if pd.notna(x) and x >= 0 else np.nan) # Price-like margin params non-negative or NaN
                    # margin_share and percental_margin_share can be negative if that's a business logic, or clip at 0. Assuming can be any float for now.

            for col in data.columns: # Price and Volume per year
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
                    data = data[data['project_type'].isin(valid_types)].copy() # Keep only valid types
            else: # Should not happen due to earlier check, but as safeguard:
                return None, "Critical error: 'project_type' column missing despite initial check passing.", available_years, [], []

            cols_to_keep = core_cols_std[:] + margin_cols_std[:] # Add margin cols to keep
            for col in optional_cols_std:
                if col in data.columns:
                    cols_to_keep.append(col)
            cols_to_keep.extend(year_data_cols_found)
            data = data[list(set(cols_to_keep))] # Use set to ensure unique columns

            # Standardize column names for display (with spaces)
            final_rename_map = {
                'project_name': 'project name', 'project_type': 'project type', 'priority': 'priority',
                'description': 'Description', 'project_link': 'Project Link',
                'base_price': 'base price', 'threshold_price': 'threshold price',
                'margin_share': 'margin share', 'fixed_purchase_price': 'fixed purchase price',
                'percental_margin_share': 'percental margin share'
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
            if invalid_types_found: st.sidebar.warning(f"Ignored rows with invalid project types: {', '.join(invalid_types_found)}. Valid types are 'reduction', 'technical removal', 'natural removal'.")
            if error_msg:
                st.sidebar.error(error_msg); st.session_state.data_loaded_successfully = False; st.session_state.working_data_full = None
                st.session_state.project_names = []; st.session_state.available_years_in_data = []; st.session_state.selected_projects = []; st.session_state.annual_targets = {}
            else:
                st.session_state.project_names = project_names_list; st.session_state.available_years_in_data = available_years_in_data
                st.session_state.working_data_full = data; st.session_state.data_loaded_successfully = True; st.sidebar.success("Data loaded successfully!")
                current_selection = st.session_state.get('selected_projects', []); valid_current_selection = [p for p in current_selection if p in project_names_list]
                if not valid_current_selection and project_names_list: st.session_state.selected_projects = project_names_list # Default to all if current selection invalid
                else: st.session_state.selected_projects = valid_current_selection
                st.session_state.annual_targets = {} # Reset annual targets on new data load
        except Exception as e:
            st.sidebar.error(f"An unexpected error occurred during file processing: {e}"); st.sidebar.error(f"Traceback: {traceback.format_exc()}")
            st.session_state.data_loaded_successfully = False; st.session_state.working_data_full = None; st.session_state.project_names = []; st.session_state.available_years_in_data = []; st.session_state.selected_projects = []; st.session_state.annual_targets = {}

    # --- Settings (Displayed only if data loaded successfully) ---
    if st.session_state.get('data_loaded_successfully', False):
        data_for_ui = st.session_state.working_data_full
        available_years_in_data = st.session_state.available_years_in_data
        project_names_list = st.session_state.project_names
        if not available_years_in_data: # This check is crucial
            st.sidebar.warning("No usable year data (price_YYYY/volume_YYYY columns) found in the uploaded file. Cannot proceed with settings.")
        else:
            st.markdown("## 2. Portfolio Settings")
            min_year_data = min(available_years_in_data)
            max_year_data = max(available_years_in_data)

            max_possible_years_to_plan = max(1, max_year_data - min_year_data + 1)
            try: current_years_to_plan_val = int(st.session_state.get('years_slider_sidebar', 5))
            except (ValueError, TypeError): current_years_to_plan_val = 5
            current_years_to_plan_val = max(1, min(current_years_to_plan_val, max_possible_years_to_plan))

            years_to_plan = st.number_input(
                label=f"Years to Plan (Starting {min_year_data})", min_value=1, max_value=max_possible_years_to_plan,
                value=current_years_to_plan_val, step=1, key='years_slider_sidebar_widget',
                help=f"Enter the number of years for portfolio planning, from 1 to {max_possible_years_to_plan} based on your data."
            )
            st.session_state.years_slider_sidebar = years_to_plan

            start_year_selected = min_year_data
            end_year_selected = start_year_selected + years_to_plan - 1
            selected_years_range = list(range(start_year_selected, end_year_selected + 1))
            actual_years_present_in_data = []
            for year_iter in selected_years_range:
                price_col_check = f"price {year_iter}"; vol_col_check = f"available volume {year_iter}" # Renamed price_col, vol_col
                if price_col_check in data_for_ui.columns and vol_col_check in data_for_ui.columns:
                    actual_years_present_in_data.append(year_iter)
            st.session_state.selected_years = actual_years_present_in_data

            if not st.session_state.selected_years:
                st.sidebar.error(f"No data available for the selected period ({start_year_selected}-{end_year_selected}). Adjust 'Years to Plan' or check CSV data.")
                st.session_state.actual_start_year = None; st.session_state.actual_end_year = None
            else:
                st.session_state.actual_start_year = min(st.session_state.selected_years)
                st.session_state.actual_end_year = max(st.session_state.selected_years)
                st.sidebar.markdown(f"Planning Horizon: **{st.session_state.actual_start_year} - {st.session_state.actual_end_year}**")
                st.session_state.constraint_type = st.radio("Constraint Type:", ('Volume', 'Budget'), index=['Volume', 'Budget'].index(st.session_state.get('constraint_type', 'Volume')), key='constraint_type_sidebar_widget', horizontal=True, help="Choose whether annual targets are defined in tons (Volume) or currency (Budget).")
                constraint_type = st.session_state.constraint_type

                st.markdown("### Annual Target Settings")
                master_target_value = st.session_state.get('master_target')

                if constraint_type == 'Volume':
                    default_val_vol = 1000
                    if master_target_value is not None : # Check if master_target_value is not None
                        try: default_val_vol = int(float(master_target_value))
                        except (ValueError, TypeError): pass # keep default if conversion fails
                    default_target = st.number_input(
                        "Default Annual Target Volume (t):", min_value=0, step=100, value=default_val_vol,
                        key='master_volume_sidebar', help="Set a default target volume per year. You can override specific years below."
                    )
                else: # Budget constraint
                    default_val_bud = 100000.0
                    if master_target_value is not None : # Check if master_target_value is not None
                        try: default_val_bud = float(master_target_value)
                        except (ValueError, TypeError): pass # keep default if conversion fails
                    default_target = st.number_input(
                        "Default Annual Target Budget (€):", min_value=0.0, step=1000.0, value=default_val_bud, format="%.2f",
                        key='master_budget_sidebar', help="Set a default target budget per year. You can override specific years below."
                    )
                st.session_state.master_target = default_target

                with st.expander("Customize Annual Targets (+/-)", expanded=False):
                    current_annual_targets = st.session_state.get('annual_targets', {})
                    updated_targets_from_inputs = {}

                    if not st.session_state.selected_years:
                        st.caption("Select years using the 'Years to Plan' input above first.")
                    else:
                        for year_val_input in st.session_state.selected_years:
                            year_target_value = current_annual_targets.get(year_val_input, default_target)
                            input_key = f"target_{year_val_input}_{constraint_type}" # Key includes constraint type
                            label = f"Target {year_val_input} [t]" if constraint_type == 'Volume' else f"Target {year_val_input} [€]"

                            if constraint_type == 'Volume':
                                try: input_val = int(float(year_target_value)) # Attempt float conversion first for robustness
                                except (ValueError, TypeError): input_val = int(default_target)
                                updated_targets_from_inputs[year_val_input] = st.number_input(
                                    label, min_value=0, step=100, value=input_val, key=input_key
                                )
                            else: # Budget
                                try: input_val = float(year_target_value)
                                except (ValueError, TypeError): input_val = float(default_target)
                                updated_targets_from_inputs[year_val_input] = st.number_input(
                                    label, min_value=0.0, step=1000.0, value=input_val, format="%.2f", key=input_key
                                )
                    st.session_state.annual_targets = updated_targets_from_inputs
                
                st.sidebar.markdown("### Allocation Goal & Preferences")
                min_fulfill = st.sidebar.slider(f"Min. Target Fulfillment (%)", 50, 100, st.session_state.get('min_fulfillment_perc', 95), help=f"Attempt >= this % of target {constraint_type} via adjustment.", key='min_fulfill_perc_sidebar')
                st.session_state.min_fulfillment_perc = min_fulfill
                min_chunk_val = st.sidebar.number_input("Min. Allocation Unit (t)", 1, step=1, value=st.session_state.get('min_alloc_chunk', 1), help="Smallest amount (tons) to allocate per project/year.", key='min_alloc_chunk_sidebar')
                st.session_state.min_alloc_chunk = int(min_chunk_val) if pd.notna(min_chunk_val) else 1

                st.sidebar.markdown("### Removal Volume Transition (If 'Reduction' Projects Used)")
                reduction_present = 'reduction' in data_for_ui['project type'].unique() if 'project type' in data_for_ui else False
                if reduction_present: st.sidebar.info("Transition settings apply if 'Reduction' projects selected.")
                else: st.sidebar.info("Transition inactive: No 'Reduction' projects found or 'project type' column missing.")
                
                removal_help_end_year = st.session_state.actual_end_year if st.session_state.actual_end_year else "end year"
                removal_help = f"Target % vol from Removals in final year ({removal_help_end_year}). Guides mix if 'Reduction' selected."
                
                try: rem_target_slider_default = int(float(st.session_state.get('removal_target_end_year', 0.8)) * 100)
                except: rem_target_slider_default = 80

                rem_target_slider = st.sidebar.slider(f"Target Removal Vol % ({removal_help_end_year})", 0, 100, rem_target_slider_default, help=removal_help, key='removal_perc_slider_sidebar', disabled=not reduction_present)
                st.session_state.removal_target_end_year = rem_target_slider / 100.0
                st.session_state.transition_speed = st.sidebar.slider("Transition Speed", 1, 10, st.session_state.get('transition_speed', 5), help="Ramp-up speed (1=Slow, 10=Fast) if Reductions selected.", key='transition_speed_slider_sidebar', disabled=not reduction_present)
                
                st.sidebar.markdown("### Removal Category Preference")
                removal_types_present = any(pt in data_for_ui['project type'].unique() for pt in ['technical removal', 'natural removal']) if 'project type' in data_for_ui else False
                rem_pref_val = st.sidebar.slider("Technical vs Natural Preference", 1, 10, st.session_state.get('removal_preference_slider', 5), format="%d", key='removal_pref_slider_sidebar', help="1 leans Natural, 5 balanced, 10 leans Technical.", disabled=not removal_types_present)
                st.session_state['removal_preference_slider'] = rem_pref_val
                tech_pref_ratio = (rem_pref_val - 1) / 9.0
                st.session_state.category_split = {'technical removal': tech_pref_ratio, 'natural removal': 1.0 - tech_pref_ratio}
                
                st.sidebar.markdown("## 3. Select Projects")
                if not project_names_list: st.sidebar.warning("No projects available.")
                else:
                    st.session_state.selected_projects = st.sidebar.multiselect("Select projects to include:", options=project_names_list, default=st.session_state.get('selected_projects', project_names_list), key='project_selector_sidebar')
                    if 'priority' in data_for_ui.columns:
                        boost_options = [p for p in project_names_list if p in st.session_state.selected_projects]
                        if boost_options:
                            current_favorite = st.session_state.get('favorite_projects_selection', [])
                            valid_default_favorite = [f for f in current_favorite if f in boost_options][:1] # Ensure only one max
                            st.session_state.favorite_projects_selection = st.sidebar.multiselect("Favorite Project (Priority Boost):", options=boost_options, default=valid_default_favorite, key='favorite_selector_sidebar', max_selections=1, help="Boost priority for one project.")
                        else: st.sidebar.info("Select projects first to enable boost."); st.session_state.favorite_projects_selection = []
                    else: st.sidebar.info("Boost disabled: No 'priority' column."); st.session_state.favorite_projects_selection = []


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
    available_years_main_display = st.session_state.available_years_in_data
    price_cols_available = [f"price {year_pd_main}" for year_pd_main in available_years_main_display if f"price {year_pd_main}" in project_df_display.columns]

    if price_cols_available:
        project_df_display['Average Price'] = project_df_display[price_cols_available].mean(axis=1, skipna=True).fillna(0.0)
    else:
        project_df_display['Average Price'] = 0.0

    with st.expander("View Project Details"):
        if not project_df_display.empty:
            display_cols_options = ['project name', 'project type', 'Average Price']
            column_config = {"Average Price": st.column_config.NumberColumn("Avg Price (€/t)", help="Average price across all years available in the input data.", format="€%.2f")}
            if 'Project Link' in project_df_display.columns:
                display_cols_options.append('Project Link')
                column_config["Project Link"] = st.column_config.LinkColumn("Project Link", display_text="Visit ->", help="Click link to visit project page (if available)")
            st.dataframe(project_df_display[display_cols_options], column_config=column_config, hide_index=True, use_container_width=True)
        else: st.write("Project data is loaded but appears to be empty or filtered out.")
    st.markdown("---")

if st.session_state.get('data_loaded_successfully', False):
    if not st.session_state.get('selected_projects'):
        st.warning("⚠️ Please select the projects you want to include in the portfolio using the sidebar (Section 3).")
    elif not st.session_state.get('selected_years'):
        st.warning("⚠️ No valid years identified for the selected planning horizon. Please adjust the 'Years to Plan' (Section 2) or verify the year columns in your CSV data.")
    else:
        required_keys = ['working_data_full', 'selected_projects', 'selected_years','actual_start_year', 'actual_end_year', 'constraint_type', 'annual_targets', 'removal_target_end_year', 'transition_speed', 'category_split', 'favorite_projects_selection', 'min_fulfillment_perc', 'min_alloc_chunk']
        keys_present = all(k in st.session_state and st.session_state.get(k) is not None for k in required_keys if k not in ['annual_targets', 'favorite_projects_selection', 'actual_start_year', 'actual_end_year']) # these can be None or {}
        keys_present = keys_present and ('annual_targets' in st.session_state) and ('favorite_projects_selection' in st.session_state)
        keys_present = keys_present and st.session_state.actual_start_year is not None and st.session_state.actual_end_year is not None


        if keys_present:
            try:
                fav_proj = st.session_state.favorite_projects_selection[0] if st.session_state.favorite_projects_selection else None
                constraint = st.session_state.constraint_type
                min_chunk = st.session_state.min_alloc_chunk
                annual_targets_to_use = st.session_state.annual_targets
                
                if not annual_targets_to_use and st.session_state.selected_years :
                    st.warning("Annual targets are not set. Please configure them in Sidebar Section 2 under 'Annual Target Settings'. Using 0 for all years.")
                    annual_targets_to_use = {yr_target: 0 for yr_target in st.session_state.selected_years} # Renamed yr

                if constraint == 'Budget': st.info(f"**Budget Mode:** Projects get budget via weighted priority/price. May get 0 vol if budget < cost of {min_chunk}t. Adjustment step may add later.")
                st.success(f"**Allocation Goal:** Attempting ≥ **{st.session_state.min_fulfillment_perc}%** of annual target {constraint}.")
                
                with st.spinner("Calculating portfolio allocation... Please wait."):
                    results, summary_df = allocate_portfolio( # Renamed summary to summary_df
                        project_data=st.session_state.working_data_full, selected_project_names=st.session_state.selected_projects,
                        selected_years=st.session_state.selected_years, start_year_portfolio=st.session_state.actual_start_year,
                        end_year_portfolio=st.session_state.actual_end_year, constraint_type=constraint, annual_targets=annual_targets_to_use,
                        removal_target_percent_end_year=st.session_state.removal_target_end_year, transition_speed=st.session_state.transition_speed,
                        category_split=st.session_state.category_split, favorite_project=fav_proj, priority_boost_percent=10,
                        min_target_fulfillment_percent=st.session_state.min_fulfillment_perc / 100.0, min_allocation_chunk=min_chunk)
                
                details_list = []
                if results:
                    for year_res, projects_res in results.items(): # Renamed year, projects
                        if projects_res:
                            for proj_res in projects_res: # Renamed proj
                                if isinstance(proj_res, dict) and (proj_res.get('allocated_volume', 0) >= min_chunk or proj_res.get('allocated_cost', 0) > 1e-6):
                                    details_list.append({
                                        'year': year_res, 'project name': proj_res.get('project name'),
                                        'type': proj_res.get('type'), 'volume': proj_res.get('allocated_volume', 0),
                                        'price': proj_res.get('price_used', None), # 'price_used' is "price YEAR"
                                        'cost': proj_res.get('allocated_cost', 0.0)
                                    })
                
                details_df = pd.DataFrame(details_list)
                
                # --- NEW: Calculate and Add Margins ---
                if not details_df.empty:
                    details_df_with_margins = add_margins_to_details_df(details_df.copy(), st.session_state.working_data_full)
                    total_portfolio_margin = details_df_with_margins['margin'].sum()
                    # Add total yearly margin to summary_df
                    if not summary_df.empty:
                        yearly_margins = details_df_with_margins.groupby('year')['margin'].sum().reset_index().rename(columns={'margin': 'Total Yearly Margin', 'year': 'Year'})
                        summary_df = pd.merge(summary_df, yearly_margins, on='Year', how='left')
                        summary_df['Total Yearly Margin'] = summary_df['Total Yearly Margin'].fillna(0.0)

                else:
                    details_df_with_margins = details_df.copy() # Still create it, but it will be empty
                    details_df_with_margins['margin'] = 0.0 # ensure column exists
                    total_portfolio_margin = 0.0
                    if not summary_df.empty:
                        summary_df['Total Yearly Margin'] = 0.0


                st.markdown("## Portfolio Summary"); col_l, col_m_new, col_r = st.columns([1.5, 1.5, 1.2], gap="large") # Adjusted columns for new metric
                with col_l:
                    st.markdown("#### Key Metrics (Overall)")
                    if not summary_df.empty:
                        total_cost_all_years = summary_df['Allocated Cost'].sum()
                        total_volume_all_years = summary_df['Allocated Volume'].sum()
                        overall_avg_price = total_cost_all_years / total_volume_all_years if total_volume_all_years > 0 else 0.0
                        st.markdown(f"""<div class="metric-box"><b>Total Portfolio Cost</b> € {total_cost_all_years:,.2f}</div>""", unsafe_allow_html=True)
                        st.markdown(f"""<div class="metric-box"><b>Total Portfolio Volume</b> {total_volume_all_years:,.0f} t</div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="metric-box"><b>Total Portfolio Cost</b> - </div>""", unsafe_allow_html=True)
                        st.markdown(f"""<div class="metric-box"><b>Total Portfolio Volume</b> - </div>""", unsafe_allow_html=True)
                
                with col_m_new: # New column for margin and avg price
                    st.markdown("#### &nbsp;") # Placeholder for alignment or another title
                    if not summary_df.empty:
                        st.markdown(f"""<div class="metric-box"><b>Total Portfolio Margin</b> € {total_portfolio_margin:,.2f}</div>""", unsafe_allow_html=True)
                        st.markdown(f"""<div class="metric-box"><b>Overall Average Price</b> € {overall_avg_price:,.2f} /t</div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="metric-box"><b>Total Portfolio Margin</b> - </div>""", unsafe_allow_html=True)
                        st.markdown(f"""<div class="metric-box"><b>Overall Average Price</b> - </div>""", unsafe_allow_html=True)


                with col_r:
                    st.markdown("#### Volume by Project Type")
                    if not details_df_with_margins.empty: # Use df with margins for consistency, though pie is on volume
                        pie_data = details_df_with_margins.groupby('type')['volume'].sum().reset_index()
                        pie_data = pie_data[pie_data['volume'] > 1e-6]
                        if not pie_data.empty:
                            st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
                            fig_pie = px.pie(pie_data, values='volume', names='type', color='type', color_discrete_map=type_color_map)
                            fig_pie.update_layout(showlegend=True, legend_title_text='Project Type', legend_orientation="h", legend_yanchor="bottom", legend_y=-0.2, legend_xanchor="center", legend_x=0.5, margin=dict(t=5, b=50, l=0, r=0), height=350) # Increased height for legend
                            fig_pie.update_traces(textposition='inside', textinfo='percent', sort=False, hole=.3, marker=dict(line=dict(color='#FFFFFF', width=1)))
                            st.plotly_chart(fig_pie, use_container_width=True); st.markdown("</div>", unsafe_allow_html=True)
                        else: st.caption("No significant volume allocated for pie chart.")
                    else: st.caption("No allocation details for pie chart.")
                
                st.markdown("---")
                if details_df_with_margins.empty and summary_df.empty: st.warning("No allocation data generated to display plots or tables.")
                elif details_df_with_margins.empty: st.warning("No detailed project allocations for composition plot.")
                else:
                    st.markdown("### Portfolio Composition & Price Over Time")
                    # Ensure 'year' is int for plotting if it comes from dict keys
                    details_df_with_margins['year'] = details_df_with_margins['year'].astype(int)
                    summary_plot_data = details_df_with_margins.groupby(['year', 'type']).agg(volume=('volume', 'sum'), cost=('cost', 'sum'), margin=('margin', 'sum')).reset_index()
                    price_summary_data = pd.DataFrame()
                    if not summary_df.empty: price_summary_data = summary_df[['Year', 'Avg. Price']].rename(columns={'Year':'year', 'Avg. Price':'avg_price'})
                    
                    fig_composition = make_subplots(specs=[[{"secondary_y": True}]])
                    y_metric = 'volume' if constraint == 'Volume' else 'cost'
                    y_label = 'Allocated Volume (t)' if constraint == 'Volume' else 'Allocated Cost (€)'
                    y_format = '{:,.0f}' if constraint == 'Volume' else '€{:,.2f}'
                    y_hover_label = 'Volume' if constraint == 'Volume' else 'Cost'
                    type_order = ['reduction', 'natural removal', 'technical removal']
                    types_in_results = details_df_with_margins['type'].unique()

                    for t_name in type_order:
                        if t_name in types_in_results:
                            df_type = summary_plot_data[summary_plot_data['type'] == t_name]
                            if not df_type.empty and y_metric in df_type.columns and df_type[y_metric].sum() > 1e-6 :
                                fig_composition.add_trace(go.Bar(x=df_type['year'], y=df_type[y_metric], name=t_name.replace('_', ' ').capitalize(), marker_color=type_color_map.get(t_name, default_color), hovertemplate=f'Year: %{{x}}<br>Type: {t_name.replace("_", " ").capitalize()}<br>{y_hover_label}: %{{y:{y_format}}}<extra></extra>'), secondary_y=False)
                    
                    if not price_summary_data.empty and 'avg_price' in price_summary_data.columns:
                        fig_composition.add_trace(go.Scatter(x=price_summary_data['year'], y=price_summary_data['avg_price'], name='Avg Price (€/t)', mode='lines+markers', marker=dict(symbol='circle', size=8), line=dict(color='#1B5E20', width=3), hovertemplate='Year: %{x}<br>Avg Price: €%{y:,.2f}/t<extra></extra>'), secondary_y=True)
                    if not summary_df.empty and 'Actual Removal Vol %' in summary_df.columns:
                        fig_composition.add_trace(go.Scatter(x=summary_df['Year'], y=summary_df['Actual Removal Vol %'], name='Actual Removal Vol %', mode='lines+markers', line=dict(color='darkorange', dash='dash'), marker=dict(symbol='star', size=8), hovertemplate='Year: %{x}<br>Actual Removal: %{y:.1f}%<extra></extra>'), secondary_y=True)
                    
                    y2_max_val = 105.0
                    if not price_summary_data.empty and 'avg_price' in price_summary_data.columns and not price_summary_data['avg_price'].empty:
                        y2_max_val = max(y2_max_val, price_summary_data['avg_price'].max() * 1.1 if pd.notna(price_summary_data['avg_price'].max()) else y2_max_val)
                    if not summary_df.empty and 'Actual Removal Vol %' in summary_df.columns and not summary_df['Actual Removal Vol %'].empty:
                         y2_max_val = max(y2_max_val, summary_df['Actual Removal Vol %'].max() * 1.1 if pd.notna(summary_df['Actual Removal Vol %'].max()) else y2_max_val)
                    
                    fig_composition.update_layout(xaxis_title='Year', yaxis_title=y_label, yaxis2_title='Avg Price (€/t) / Actual Removal %', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), barmode='stack', template="plotly_white", margin=dict(t=20, l=0, r=0, b=0), yaxis=dict(rangemode='tozero'), yaxis2=dict(rangemode='tozero', range=[0, y2_max_val]), hovermode="x unified")
                    if st.session_state.selected_years: fig_composition.update_xaxes(tickmode='array', tickvals=st.session_state.selected_years, dtick=1)
                    st.plotly_chart(fig_composition, use_container_width=True)

                st.markdown("### Detailed Allocation by Project and Year")
                pivot_display = pd.DataFrame() # Initialize
                if not details_df_with_margins.empty or not summary_df.empty:
                    try:
                        pivot_final = pd.DataFrame()
                        years_present_in_results_pivot = st.session_state.selected_years[:] # Use a copy

                        if not details_df_with_margins.empty:
                            details_df_with_margins['year'] = pd.to_numeric(details_df_with_margins['year'])
                            pivot_intermediate = pd.pivot_table(details_df_with_margins,
                                                                values=['volume', 'cost', 'price', 'margin'], # ADDED margin
                                                                index=['project name', 'type'], columns='year',
                                                                aggfunc={'volume': 'sum', 'cost': 'sum', 'price': 'first', 'margin': 'sum'}) # ADDED margin agg
                            if not pivot_intermediate.empty:
                                pivot_final = pivot_intermediate.swaplevel(0, 1, axis=1)
                                metric_order = ['volume', 'cost', 'price', 'margin'] # ADDED margin
                                
                                # Ensure years_present_in_results_pivot only contains years actually in pivot_final columns
                                years_in_pivot_cols = sorted([yr_piv for yr_piv in pivot_final.columns.get_level_values(0).unique() if isinstance(yr_piv, (int, np.integer, float, np.floating))])
                                years_present_in_results_pivot = [yr for yr in years_present_in_results_pivot if yr in years_in_pivot_cols]


                                if years_present_in_results_pivot: # only proceed if there are valid years
                                    final_multi_index = pd.MultiIndex.from_product([years_present_in_results_pivot, metric_order], names=['year', 'metric'])
                                    pivot_final = pivot_final.reindex(columns=final_multi_index).sort_index(axis=1, level=[0, 1])
                                    pivot_final.index.names = ['Project Name', 'Type']
                                else: # No common years or empty pivot from start
                                    pivot_final = pd.DataFrame(index=pivot_intermediate.index if not pivot_intermediate.empty else None) # Keep index if possible

                        total_data_dict = {} # Renamed total_data to total_data_dict
                        if not summary_df.empty and years_present_in_results_pivot: # Check years_present_in_results_pivot
                            summary_indexed = summary_df.set_index('Year')
                            for year_sum_detail in years_present_in_results_pivot: # Renamed year to avoid conflict
                                if year_sum_detail in summary_indexed.index:
                                    vol, cost, avg_price, yearly_margin = summary_indexed.loc[year_sum_detail, ['Allocated Volume', 'Allocated Cost', 'Avg. Price', 'Total Yearly Margin']]
                                else: vol, cost, avg_price, yearly_margin = 0, 0.0, 0.0, 0.0
                                total_data_dict[(year_sum_detail, 'volume')] = vol
                                total_data_dict[(year_sum_detail, 'cost')] = cost
                                total_data_dict[(year_sum_detail, 'price')] = avg_price
                                total_data_dict[(year_sum_detail, 'margin')] = yearly_margin # ADDED margin for total row

                        total_row_index = pd.MultiIndex.from_tuples([('Total Portfolio', 'All Types')], names=['Project Name', 'Type'])
                        total_row_df = pd.DataFrame(total_data_dict, index=total_row_index)
                        
                        if pivot_final.empty and not total_row_df.empty : pivot_display = total_row_df
                        elif not pivot_final.empty and not total_row_df.empty : pivot_display = pd.concat([pivot_final, total_row_df])
                        elif not pivot_final.empty : pivot_display = pivot_final
                        # else pivot_display remains empty DataFrame

                        if not pivot_display.empty:
                            # Calculate "Total Margin" column per project (sum of yearly margins)
                            # Ensure 'margin' columns exist before trying to sum them
                            margin_columns_exist = any(col_name[1] == 'margin' for col_name in pivot_display.columns if isinstance(col_name, tuple) and len(col_name)==2)
                            if margin_columns_exist:
                                pivot_display['Total Margin'] = pivot_display.xs('margin', axis=1, level='metric').sum(axis=1)
                            else: # If no margin columns (e.g. all years had no margin data)
                                pivot_display['Total Margin'] = 0.0


                            pivot_display = pivot_display.fillna(0)
                            formatter = {}
                            for col_tuple in pivot_display.columns: # Iterate through columns directly
                                if isinstance(col_tuple, tuple) and len(col_tuple) == 2: # Yearly metrics
                                    year_col_val, metric_col_val = col_tuple
                                    if metric_col_val == 'volume': formatter[col_tuple] = '{:,.0f} t'
                                    elif metric_col_val == 'cost': formatter[col_tuple] = '€{:,.2f}'
                                    elif metric_col_val == 'price': formatter[col_tuple] = lambda x_val: f'€{x_val:,.2f}/t' if pd.notna(x_val) and x_val != 0 else '-'
                                    elif metric_col_val == 'margin': formatter[col_tuple] = '€{:,.2f}' # ADDED margin format
                                elif col_tuple == 'Total Margin': # For the single 'Total Margin' column
                                     formatter[col_tuple] = '€{:,.2f}'

                            st.dataframe(pivot_display.style.format(formatter, na_rep="-"), use_container_width=True)
                        else: st.info("No data for detailed allocation table after processing.")
                    except Exception as e: st.error(f"Could not create detailed allocation table: {e}"); st.error(f"Traceback: {traceback.format_exc()}")
                else: st.info("No allocation details or summary data to display.")
                
                if not pivot_display.empty:
                    csv_df = pivot_display.copy()
                    new_cols = []
                    for col_item in csv_df.columns.values:
                        if isinstance(col_item, tuple): new_cols.append(f"{str(col_item[0])}_{col_item[1]}")
                        else: new_cols.append(str(col_item)) # For 'Total Margin'
                    csv_df.columns = new_cols
                    csv_df = csv_df.reset_index()
                    try:
                        csv_string = csv_df.to_csv(index=False).encode('utf-8')
                        st.markdown("---")
                        st.download_button(label="Download Detailed Allocation (CSV)", data=csv_string, file_name=f"portfolio_allocation_{datetime.date.today()}.csv", mime='text/csv', key='download-csv')
                    except Exception as e_csv: st.error(f"Error generating CSV for download: {e_csv}")

            except ValueError as e_val: st.error(f"Configuration or Allocation Error: {e_val}")
            except KeyError as e_key: st.error(f"Data Error: Missing key: '{e_key}'. Check CSV format/names & selections. Full Trace: {traceback.format_exc()}")
            except Exception as e_gen: st.error(f"Unexpected error during portfolio generation: {e_gen}"); st.error(f"Traceback: {traceback.format_exc()}")
        else: st.error("Missing required settings. Please ensure data is loaded and all sidebar settings are configured correctly (especially planning horizon).")

# --- Footer ---
st.markdown("---")
try:
    zurich_tz = pytz.timezone('Europe/Zurich')
    now_zurich = datetime.datetime.now(zurich_tz)
    st.caption(f"Report generated: {now_zurich.strftime('%Y-%m-%d %H:%M:%S %Z')}")
except Exception: # Catch generic exception for timezone issues
    now_local = datetime.datetime.now()
    st.caption(f"Report generated: {now_local.strftime('%Y-%m-%d %H:%M:%S')} (Timezone: Server Local)")
