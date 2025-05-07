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
    min_allocation_chunk: int = 1
) -> tuple[dict, pd.DataFrame]:
    portfolio_details = {year: [] for year in selected_years}
    yearly_summary_list = []

    if not selected_project_names:
        st.warning("No projects selected for allocation.")
        return {}, pd.DataFrame(columns=['Year', f'Target {constraint_type}', 'Allocated Volume', 'Allocated Cost', 'Avg. Price', 'Actual Removal Vol %', 'Target Removal Vol %', 'Total Yearly Margin'])

    project_data_selected = project_data[project_data['project name'].isin(selected_project_names)].copy()

    if project_data_selected.empty:
        st.warning("Selected projects not found in the provided data.")
        return {}, pd.DataFrame(columns=['Year', f'Target {constraint_type}', 'Allocated Volume', 'Allocated Cost', 'Avg. Price', 'Actual Removal Vol %', 'Target Removal Vol %', 'Total Yearly Margin'])

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
        if f"price {year}" not in project_data_selected.columns: missing_years_data.append(f"price {year}")
        if f"available volume {year}" not in project_data_selected.columns: missing_years_data.append(f"available volume {year}")

    if missing_years_data:
        years_affected = sorted(list(set(int(col.split()[-1]) for col in missing_years_data if col.split()[-1].isdigit())))
        raise ValueError(f"Input data is missing price/volume information for required year(s) for allocation: {', '.join(map(str, years_affected))}.")

    numeric_cols_to_check = ['priority'] + price_cols_needed + volume_cols_needed
    for col in numeric_cols_to_check:
        if col in project_data_selected.columns:
            project_data_selected[col] = pd.to_numeric(project_data_selected[col], errors='coerce')
            if col == 'priority': project_data_selected[col] = project_data_selected[col].fillna(0)
            elif col.startswith("available volume"): project_data_selected[col] = project_data_selected[col].fillna(0).apply(lambda x: int(x) if pd.notna(x) and x >= 0 else 0).clip(lower=0)
            elif col.startswith("price"): project_data_selected[col] = project_data_selected[col].fillna(0.0).apply(lambda x: float(x) if pd.notna(x) and x >= 0 else 0.0).clip(lower=0.0)

    all_project_types_in_selection = project_data_selected['project type'].unique()
    is_reduction_selected = 'reduction' in all_project_types_in_selection
    total_years_duration = end_year_portfolio - start_year_portfolio

    for year_loop_var in selected_years: # Renamed 'year' to avoid conflict in deeper scopes
        yearly_target = annual_targets.get(year_loop_var, 0)
        price_col = f"price {year_loop_var}"
        volume_col = f"available volume {year_loop_var}"

        year_total_allocated_vol = 0
        year_total_allocated_cost = 0.0
        summary_template = {
            'Year': year_loop_var, f'Target {constraint_type}': yearly_target, 'Allocated Volume': 0,
            'Allocated Cost': 0.0, 'Avg. Price': 0.0, 'Actual Removal Vol %': 0.0,
            'Target Removal Vol %': 0.0, 'Total Yearly Margin': 0.0
        }

        if yearly_target <= 0:
            yearly_summary_list.append(summary_template); portfolio_details[year_loop_var] = []; continue

        target_percentages = {}
        if is_reduction_selected:
            start_removal_percent = 0.10; end_removal_percent = removal_target_percent_end_year
            progress = 1.0 if total_years_duration <= 0 else max(0, min(1, (year_loop_var - start_year_portfolio) / total_years_duration))
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
                target_percentages['technical removal'] = (tech_alloc_share / total_alloc_share) if tech_selected else 0.0
                target_percentages['natural removal'] = (nat_alloc_share / total_alloc_share) if nat_selected else 0.0
            elif tech_selected or nat_selected:
                 target_percentages['technical removal'] = 1.0 if tech_selected and not nat_selected else (0.5 if tech_selected and nat_selected else 0.0)
                 target_percentages['natural removal'] = 1.0 if nat_selected and not tech_selected else (0.5 if tech_selected and nat_selected else 0.0)
            else:
                 target_percentages['technical removal'] = 0.0; target_percentages['natural removal'] = 0.0
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
            yearly_summary_list.append(summary_template); portfolio_details[year_loop_var] = []; continue

        for project_type_alloc_loop in all_project_types_in_selection:
            target_share = target_percentages.get(project_type_alloc_loop, 0)
            if target_share <= 0: continue
            target_resource = yearly_target * target_share
            projects_of_type = projects_year_df[projects_year_df['project type'] == project_type_alloc_loop].copy()
            if projects_of_type.empty: continue

            total_priority_in_type = projects_of_type['priority'].sum()
            if total_priority_in_type <= 0:
                num_projects_in_type = len(projects_of_type)
                projects_of_type['norm_prio_base'] = (1.0 / num_projects_in_type) if num_projects_in_type > 0 else 0
            else: projects_of_type['norm_prio_base'] = projects_of_type['priority'] / total_priority_in_type
            current_priorities = projects_of_type.set_index('project name')['norm_prio_base'].to_dict()
            final_priorities = current_priorities.copy()

            if favorite_project and favorite_project in final_priorities and projects_of_type.loc[projects_of_type['project name'] == favorite_project, 'project type'].iloc[0] == project_type_alloc_loop :
                fav_proj_base_prio = current_priorities[favorite_project]; boost_factor = priority_boost_percent / 100.0
                priority_increase = fav_proj_base_prio * boost_factor; new_fav_proj_prio = fav_proj_base_prio + priority_increase
                other_projects = [p for p in current_priorities if p != favorite_project]
                sum_other_priorities = sum(current_priorities[p] for p in other_projects)
                temp_priorities = {favorite_project: new_fav_proj_prio}; reduction_factor = 0
                if sum_other_priorities > 1e-9: reduction_factor = priority_increase / sum_other_priorities
                for name in other_projects: temp_priorities[name] = max(0, current_priorities[name] * (1 - reduction_factor))
                total_final_prio = sum(temp_priorities.values())
                if total_final_prio > 1e-9: final_priorities = {p: prio / total_final_prio for p, prio in temp_priorities.items()}
                elif favorite_project in temp_priorities : final_priorities = {favorite_project: 1.0}

            project_weights = {}; total_weight = 0
            if constraint_type == 'Budget':
                for _, row_budget_loop in projects_of_type.iterrows():
                    name_budget = row_budget_loop['project name']; final_prio_budget = final_priorities.get(name_budget, 0); price_budget = row_budget_loop[price_col]
                    weight = final_prio_budget * price_budget if price_budget > 0 else 0; project_weights[name_budget] = weight; total_weight += weight

            for idx_alloc_loop, row_alloc_loop in projects_of_type.iterrows():
                name_alloc = row_alloc_loop['project name']; final_prio_alloc = final_priorities.get(name_alloc, 0); available_vol_alloc = row_alloc_loop[volume_col]; price_alloc = row_alloc_loop[price_col]
                allocated_volume = 0; allocated_cost = 0.0
                projects_year_df.loc[projects_year_df['project name'] == name_alloc, 'final_priority'] = final_prio_alloc
                if final_prio_alloc <= 0 or price_alloc <= 0 or available_vol_alloc < min_allocation_chunk: continue

                if constraint_type == 'Volume':
                    target_volume_proj = target_resource * final_prio_alloc
                    allocated_volume = min(target_volume_proj, available_vol_alloc)
                elif constraint_type == 'Budget':
                    if total_weight > 1e-9:
                        weight_normalized = project_weights.get(name_alloc, 0) / total_weight
                        target_budget_proj = target_resource * weight_normalized
                        target_volume_proj = target_budget_proj / price_alloc if price_alloc > 0 else 0
                        allocated_volume = min(target_volume_proj, available_vol_alloc)
                    elif len(projects_of_type) == 1:
                             target_volume_proj = target_resource / price_alloc if price_alloc > 0 else 0
                             allocated_volume = min(target_volume_proj, available_vol_alloc)
                    else: allocated_volume = 0
                
                allocated_volume = int(max(0, math.floor(allocated_volume / min_allocation_chunk) * min_allocation_chunk))

                if allocated_volume >= min_allocation_chunk:
                    allocated_cost = allocated_volume * price_alloc
                    # Locate the correct row in projects_year_df to update
                    project_index_in_year_df = projects_year_df[projects_year_df['project name'] == name_alloc].index
                    if not project_index_in_year_df.empty:
                        idx_to_update = project_index_in_year_df[0]
                        projects_year_df.loc[idx_to_update, 'initial_allocated_volume'] += allocated_volume
                        projects_year_df.loc[idx_to_update, 'initial_allocated_cost'] += allocated_cost
                    year_total_allocated_vol += allocated_volume; year_total_allocated_cost += allocated_cost
        
        target_threshold = yearly_target * min_target_fulfillment_percent
        current_metric_total = year_total_allocated_cost if constraint_type == 'Budget' else year_total_allocated_vol

        if current_metric_total < target_threshold and yearly_target > 0 :
            needed = target_threshold - current_metric_total
            projects_year_df['remaining_volume'] = projects_year_df[volume_col] - projects_year_df['initial_allocated_volume']
            adjustment_candidates = projects_year_df[
                (projects_year_df['remaining_volume'] >= min_allocation_chunk) &
                (projects_year_df[price_col] > 0)
            ].sort_values(by=['priority', price_col], ascending=[False, True]).copy()

            for idx_adj_loop, row_adj_loop in adjustment_candidates.iterrows():
                if needed <= (1e-2 if constraint_type=='Budget' else 0): break
                name_adj = row_adj_loop['project name']; price_adj = row_adj_loop[price_col]; available_for_adj = row_adj_loop['remaining_volume']
                volume_to_add = 0; cost_to_add = 0.0

                if constraint_type == 'Volume': add_vol = min(available_for_adj, needed)
                else: max_affordable_vol = needed / price_adj if price_adj > 0 else 0; add_vol = min(available_for_adj, max_affordable_vol)
                
                add_vol_chunked = int(math.floor(add_vol / min_allocation_chunk) * min_allocation_chunk)

                if add_vol_chunked >= min_allocation_chunk:
                    cost_increase = add_vol_chunked * price_adj
                    if constraint_type == 'Volume' or (cost_increase <= needed * 1.1 or cost_increase < price_adj * min_allocation_chunk * 1.5):
                        volume_to_add = add_vol_chunked; cost_to_add = cost_increase
                        needed -= cost_to_add if constraint_type == 'Budget' else volume_to_add
                
                if volume_to_add > 0:
                    project_index_in_year_df_adj = projects_year_df[projects_year_df['project name'] == name_adj].index
                    if not project_index_in_year_df_adj.empty:
                        idx_to_update_adj = project_index_in_year_df_adj[0]
                        projects_year_df.loc[idx_to_update_adj, 'initial_allocated_volume'] += volume_to_add
                        projects_year_df.loc[idx_to_update_adj, 'initial_allocated_cost'] += cost_to_add
                    year_total_allocated_vol += volume_to_add; year_total_allocated_cost += cost_to_add

        final_allocations_list = []
        final_year_allocations_df = projects_year_df[projects_year_df['initial_allocated_volume'] >= min_allocation_chunk].copy()
        for _, row_final_loop in final_year_allocations_df.iterrows():
            current_price = row_final_loop.get(price_col, None)
            final_allocations_list.append({
                'project name': row_final_loop['project name'], 'type': row_final_loop['project type'],
                'allocated_volume': row_final_loop['initial_allocated_volume'],
                'allocated_cost': row_final_loop['initial_allocated_cost'],
                'price_used': current_price, 'priority_applied': row_final_loop['final_priority']
            })
        portfolio_details[year_loop_var] = final_allocations_list
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
    if pd.isna(allocated_price_this_year) or allocated_price_this_year < 0: return 0.0

    base_price_col = 'base price'; threshold_price_col = 'threshold price'
    margin_share_col = 'margin share'; fixed_purchase_price_col = 'fixed purchase price'
    percental_margin_share_col = 'percental margin share'
    margin_per_unit = 0.0

    bp = project_data_row.get(base_price_col); tp = project_data_row.get(threshold_price_col)
    ms = project_data_row.get(margin_share_col)

    try:
        if pd.notna(bp) and pd.notna(tp) and pd.notna(ms):
            bp, tp, ms = float(bp), float(tp), float(ms)
            margin_from_share = max(0, allocated_price_this_year - tp) * ms
            margin_from_base_to_threshold = tp - bp
            margin_per_unit = margin_from_share + margin_from_base_to_threshold
        elif pd.notna(project_data_row.get(fixed_purchase_price_col)):
            fpp = float(project_data_row.get(fixed_purchase_price_col))
            margin_per_unit = allocated_price_this_year - fpp
        elif pd.notna(project_data_row.get(percental_margin_share_col)):
            pms_val = float(project_data_row.get(percental_margin_share_col)) # Renamed pms
            margin_per_unit = allocated_price_this_year * pms_val
    except (ValueError, TypeError): margin_per_unit = 0.0 # Fallback if conversion fails
    
    return margin_per_unit if pd.notna(margin_per_unit) and margin_per_unit > -float('inf') else 0.0

def add_margins_to_details_df(details_df: pd.DataFrame, project_master_data: pd.DataFrame) -> pd.DataFrame:
    if details_df.empty or project_master_data.empty:
        details_df['margin_per_unit'] = 0.0; details_df['margin'] = 0.0
        return details_df

    details_with_margins_list = [] # Renamed details_with_margins
    project_lookup = project_master_data.set_index('project name')

    for _, row_detail_margin in details_df.iterrows(): # Renamed row
        project_name = row_detail_margin['project name']; allocated_volume = row_detail_margin['volume']
        price_used = row_detail_margin['price']
        new_row_dict = row_detail_margin.to_dict() # Renamed new_row
        margin_val, margin_pu_val = 0.0, 0.0

        if project_name in project_lookup.index:
            project_data_row = project_lookup.loc[project_name]
            if pd.notna(price_used) and allocated_volume > 0:
                margin_pu_val = get_margin_per_unit(project_data_row, price_used)
                margin_val = margin_pu_val * allocated_volume
        
        new_row_dict['margin_per_unit'] = margin_pu_val; new_row_dict['margin'] = margin_val
        details_with_margins_list.append(new_row_dict)
    return pd.DataFrame(details_with_margins_list)

# ==================================
# Streamlit App Layout & Logic
# ==================================
with st.sidebar:
    st.markdown("## 1. Load Data")
    df_upload = st.file_uploader(
        "Upload Project Data CSV", type="csv", key="uploader_sidebar",
        help="Required: `project name`, `project type`, `priority`, `price_YYYY`, `available_volume_YYYY`. Optional margins: `base price`, `threshold price`, `margin share`, `fixed purchase price`, `percental margin share`. Optional: `description`, `project_link`."
    )
    default_values = {'working_data_full': None, 'selected_years': [], 'selected_projects': [], 'project_names': [], 'favorite_projects_selection': [], 'actual_start_year': None, 'actual_end_year': None, 'available_years_in_data': [], 'constraint_type': 'Volume', 'removal_target_end_year': 0.8, 'transition_speed': 5, 'category_split': {'technical removal': 0.5, 'natural removal': 0.5}, 'annual_targets': {}, 'master_target': None, 'data_loaded_successfully': False, 'years_slider_sidebar': 5, 'min_fulfillment_perc': 95, 'removal_preference_slider': 5, 'min_alloc_chunk': 1}
    for key, default_value in default_values.items():
        if key not in st.session_state: st.session_state[key] = default_value

    if df_upload:
        @st.cache_data
        def load_and_prepare_data(uploaded_file):
            try:
                data = pd.read_csv(uploaded_file)
                data.columns = data.columns.str.lower().str.strip().str.replace(r'\s+', '_', regex=True)
                standardized_columns_found = data.columns.tolist()
            except Exception as read_error:
                return None, f"Error reading or initially processing CSV file: {read_error}. Ensure it's a valid CSV.", [], [], []

            core_cols_std = ['project_name', 'project_type', 'priority']
            optional_cols_std = ['description', 'project_link']
            margin_cols_std = ['base_price', 'threshold_price', 'margin_share', 'fixed_purchase_price', 'percental_margin_share']

            missing_essential = [col for col in core_cols_std if col not in standardized_columns_found]
            if missing_essential:
                found_cols_str = ", ".join(standardized_columns_found)
                error_message = (
                    f"CSV is missing essential columns: {', '.join(missing_essential)}. "
                    f"The script expects these exact names after standardizing your CSV headers (lowercase, all whitespace to single underscore). "
                    f"Standardized columns FOUND in your uploaded CSV: [{found_cols_str}]. "
                    "Please carefully compare the missing list with the found list. If 'project_name', 'project_type', or 'priority' are in the 'FOUND' list but still reported as missing, there might be an extremely subtle issue. Otherwise, check your CSV headers."
                )
                return None, error_message, [], [], []

            for m_col in margin_cols_std:
                if m_col not in data.columns: data[m_col] = np.nan

            numeric_prefixes_std = ['price_', 'available_volume_']
            cols_to_convert_numeric = ['priority'] + margin_cols_std
            available_years = set(); year_data_cols_found = []

            for col in data.columns:
                for prefix in numeric_prefixes_std:
                    if col.startswith(prefix) and col[len(prefix):].isdigit():
                        cols_to_convert_numeric.append(col); year_data_cols_found.append(col)
                        available_years.add(int(col[len(prefix):])); break
            
            if not available_years: st.sidebar.warning("No columns like 'price_YYYY' or 'available_volume_YYYY' found. Allocation might not be meaningful.")

            for col in list(set(cols_to_convert_numeric)):
                if col in data.columns: data[col] = pd.to_numeric(data[col], errors='coerce')

            data['priority'] = data['priority'].fillna(0).clip(lower=0)
            for m_col in margin_cols_std:
                 if m_col in data.columns:
                    if m_col in ['base_price', 'threshold_price', 'fixed_purchase_price']:
                         data[m_col] = data[m_col].apply(lambda x: x if pd.notna(x) and x >= 0 else np.nan)

            for col in data.columns:
                if col.startswith('available_volume_') and col in year_data_cols_found: data[col] = data[col].fillna(0).apply(lambda x: max(0, int(x)) if pd.notna(x) else 0).clip(lower=0)
                elif col.startswith('price_') and col in year_data_cols_found: data[col] = data[col].fillna(0.0).apply(lambda x: max(0.0, float(x)) if pd.notna(x) else 0.0).clip(lower=0)
            
            available_years = sorted(list(available_years)) if available_years else []
            invalid_types_found = []
            if 'project_type' in data.columns:
                data['project_type'] = data['project_type'].astype(str).str.lower().str.strip()
                valid_types = ['reduction', 'technical removal', 'natural removal']
                invalid_types_df = data[~data['project_type'].isin(valid_types)]
                if not invalid_types_df.empty:
                    invalid_types_found = invalid_types_df['project_type'].unique().tolist()
                    data = data[data['project_type'].isin(valid_types)].copy()
            
            all_expected_std_cols = core_cols_std + margin_cols_std + optional_cols_std + year_data_cols_found
            cols_to_keep_final = [c for c in all_expected_std_cols if c in data.columns]
            data = data[list(set(cols_to_keep_final))]

            final_rename_map = {
                'project_name': 'project name', 'project_type': 'project type', 'priority': 'priority',
                'description': 'Description', 'project_link': 'Project Link',
                'base_price': 'base price', 'threshold_price': 'threshold price',
                'margin_share': 'margin share', 'fixed_purchase_price': 'fixed purchase price',
                'percental_margin_share': 'percental margin share'
            }
            for yr_col_std in year_data_cols_found: final_rename_map[yr_col_std] = yr_col_std.replace('_', ' ')
            
            rename_map_for_df_final = {k_std: v_disp for k_std, v_disp in final_rename_map.items() if k_std in data.columns}
            data.rename(columns=rename_map_for_df_final, inplace=True)

            project_names_list = []
            if 'project name' in data.columns: project_names_list = sorted(data['project name'].unique().tolist())
            
            return data, None, available_years, project_names_list, invalid_types_found

        try:
            data, error_msg, available_years_in_data, project_names_list, invalid_types_found = load_and_prepare_data(df_upload)
            if invalid_types_found: st.sidebar.warning(f"Ignored rows with invalid project types: {', '.join(invalid_types_found)}. Valid: 'reduction', 'technical removal', 'natural removal'.")
            if error_msg:
                st.sidebar.error(error_msg); st.session_state.data_loaded_successfully = False; st.session_state.working_data_full = None
                st.session_state.project_names = []; st.session_state.available_years_in_data = []; st.session_state.selected_projects = []; st.session_state.annual_targets = {}
            else:
                st.session_state.project_names = project_names_list; st.session_state.available_years_in_data = available_years_in_data
                st.session_state.working_data_full = data; st.session_state.data_loaded_successfully = True; st.sidebar.success("Data loaded successfully!")
                current_selection = st.session_state.get('selected_projects', []); valid_current_selection = [p for p in current_selection if p in project_names_list]
                st.session_state.selected_projects = valid_current_selection if valid_current_selection or not project_names_list else project_names_list
                st.session_state.annual_targets = {}
        except Exception as e_load:
            st.sidebar.error(f"Unexpected error during file processing: {e_load}"); st.sidebar.error(f"Traceback: {traceback.format_exc()}")
            st.session_state.data_loaded_successfully = False; st.session_state.working_data_full = None; st.session_state.project_names = []; st.session_state.available_years_in_data = []; st.session_state.selected_projects = []; st.session_state.annual_targets = {}

    if st.session_state.get('data_loaded_successfully', False):
        data_for_ui = st.session_state.working_data_full
        available_years_in_data_sb = st.session_state.available_years_in_data # Renamed
        project_names_list_sb = st.session_state.project_names # Renamed

        if not available_years_in_data_sb:
            st.sidebar.warning("No usable year data (price_YYYY/volume_YYYY columns) found. Settings disabled.")
        else:
            st.markdown("## 2. Portfolio Settings")
            min_year_data = min(available_years_in_data_sb); max_year_data = max(available_years_in_data_sb)
            max_possible_years_to_plan = max(1, max_year_data - min_year_data + 1)
            try: current_years_to_plan_val = int(st.session_state.get('years_slider_sidebar', 5))
            except: current_years_to_plan_val = 5
            current_years_to_plan_val = max(1, min(current_years_to_plan_val, max_possible_years_to_plan))

            years_to_plan = st.number_input(
                label=f"Years to Plan (Starting {min_year_data})", min_value=1, max_value=max_possible_years_to_plan,
                value=current_years_to_plan_val, step=1, key='years_slider_sidebar_widget',
                help=f"Enter # years for planning, 1 to {max_possible_years_to_plan}."
            )
            st.session_state.years_slider_sidebar = years_to_plan

            start_year_selected = min_year_data; end_year_selected = start_year_selected + years_to_plan - 1
            selected_years_range = list(range(start_year_selected, end_year_selected + 1))
            actual_years_present_in_data = [yr for yr in selected_years_range if f"price {yr}" in data_for_ui.columns and f"available volume {yr}" in data_for_ui.columns]
            st.session_state.selected_years = actual_years_present_in_data

            if not st.session_state.selected_years:
                st.sidebar.error(f"No data for period ({start_year_selected}-{end_year_selected}). Adjust 'Years to Plan'.")
                st.session_state.actual_start_year = None; st.session_state.actual_end_year = None
            else:
                st.session_state.actual_start_year = min(st.session_state.selected_years)
                st.session_state.actual_end_year = max(st.session_state.selected_years)
                st.sidebar.markdown(f"Planning Horizon: **{st.session_state.actual_start_year} - {st.session_state.actual_end_year}**")
                st.session_state.constraint_type = st.radio("Constraint Type:", ('Volume', 'Budget'), index=['Volume', 'Budget'].index(st.session_state.get('constraint_type', 'Volume')), key='constraint_type_sidebar_widget', horizontal=True)
                constraint_type_sb = st.session_state.constraint_type # Renamed

                st.markdown("### Annual Target Settings")
                master_target_value = st.session_state.get('master_target')
                default_target_val = 1000 if constraint_type_sb == 'Volume' else 100000.0 # Renamed default_target
                if master_target_value is not None:
                    try: default_target_val = float(master_target_value) if constraint_type_sb == 'Budget' else int(float(master_target_value))
                    except: pass # Keep initial default if conversion fails
                
                num_input_kwargs = {"min_value": 0.0 if constraint_type_sb == 'Budget' else 0,
                                    "step": 1000.0 if constraint_type_sb == 'Budget' else 100,
                                    "value": default_target_val,
                                    "key": f'master_{constraint_type_sb.lower()}_sidebar',
                                    "help": "Set default annual target. Override specific years below."}
                if constraint_type_sb == 'Budget': num_input_kwargs["format"] = "%.2f"
                default_target_input = st.number_input(f"Default Annual Target ({'€' if constraint_type_sb == 'Budget' else 't'}):", **num_input_kwargs)
                st.session_state.master_target = default_target_input

                with st.expander("Customize Annual Targets (+/-)", expanded=False):
                    current_annual_targets = st.session_state.get('annual_targets', {})
                    updated_targets_from_inputs = {}
                    if not st.session_state.selected_years: st.caption("Select years via 'Years to Plan'.")
                    else:
                        for year_input_sb in st.session_state.selected_years: # Renamed
                            year_target_val = current_annual_targets.get(year_input_sb, default_target_input)
                            input_key = f"target_{year_input_sb}_{constraint_type_sb}"
                            label = f"Target {year_input_sb} ({'€' if constraint_type_sb == 'Budget' else 't'})"
                            num_input_kwargs_yr = {"min_value": 0.0 if constraint_type_sb == 'Budget' else 0,
                                                "step": 1000.0 if constraint_type_sb == 'Budget' else 100,
                                                "key": input_key}
                            try: num_input_kwargs_yr["value"] = float(year_target_val) if constraint_type_sb == 'Budget' else int(float(year_target_val))
                            except: num_input_kwargs_yr["value"] = float(default_target_input) if constraint_type_sb == 'Budget' else int(default_target_input)
                            if constraint_type_sb == 'Budget': num_input_kwargs_yr["format"] = "%.2f"
                            updated_targets_from_inputs[year_input_sb] = st.number_input(label, **num_input_kwargs_yr)
                    st.session_state.annual_targets = updated_targets_from_inputs
                
                st.sidebar.markdown("### Allocation Goal & Preferences")
                st.session_state.min_fulfillment_perc = st.sidebar.slider(f"Min. Target Fulfillment (%)", 50, 100, st.session_state.get('min_fulfillment_perc', 95), key='min_fulfill_perc_sidebar')
                st.session_state.min_alloc_chunk = int(st.sidebar.number_input("Min. Allocation Unit (t)", 1, step=1, value=st.session_state.get('min_alloc_chunk', 1), key='min_alloc_chunk_sidebar') or 1)

                st.sidebar.markdown("### Removal Volume Transition (If 'Reduction' Projects Used)")
                reduction_present = 'reduction' in data_for_ui['project type'].unique() if 'project type' in data_for_ui else False
                st.sidebar.info("Transition settings apply if 'Reduction' projects are selected." if reduction_present else "Transition inactive: No 'Reduction' projects.")
                removal_help_end_year = st.session_state.actual_end_year if st.session_state.actual_end_year else "end year"
                try: rem_target_default = int(float(st.session_state.get('removal_target_end_year', 0.8)) * 100)
                except: rem_target_default = 80
                st.session_state.removal_target_end_year = st.sidebar.slider(f"Target Removal Vol % ({removal_help_end_year})", 0, 100, rem_target_default, key='removal_perc_slider_sidebar', disabled=not reduction_present) / 100.0
                st.session_state.transition_speed = st.sidebar.slider("Transition Speed", 1, 10, st.session_state.get('transition_speed', 5), key='transition_speed_slider_sidebar', disabled=not reduction_present)
                
                st.sidebar.markdown("### Removal Category Preference")
                removal_types_present = any(pt in data_for_ui['project type'].unique() for pt in ['technical removal', 'natural removal']) if 'project type' in data_for_ui else False
                rem_pref_val = st.sidebar.slider("Technical vs Natural Preference", 1, 10, st.session_state.get('removal_preference_slider', 5), format="%d", key='removal_pref_slider_sidebar', disabled=not removal_types_present)
                st.session_state['removal_preference_slider'] = rem_pref_val; tech_pref_ratio = (rem_pref_val - 1) / 9.0
                st.session_state.category_split = {'technical removal': tech_pref_ratio, 'natural removal': 1.0 - tech_pref_ratio}
                
                st.sidebar.markdown("## 3. Select Projects")
                if not project_names_list_sb: st.sidebar.warning("No projects available.")
                else:
                    st.session_state.selected_projects = st.sidebar.multiselect("Select projects to include:", options=project_names_list_sb, default=st.session_state.get('selected_projects', project_names_list_sb), key='project_selector_sidebar')
                    if 'priority' in data_for_ui.columns: # Check on display name
                        boost_options = [p for p in project_names_list_sb if p in st.session_state.selected_projects]
                        if boost_options:
                            fav_default = [f for f in st.session_state.get('favorite_projects_selection', []) if f in boost_options][:1]
                            st.session_state.favorite_projects_selection = st.sidebar.multiselect("Favorite Project (Priority Boost):", options=boost_options, default=fav_default, key='favorite_selector_sidebar', max_selections=1)
                        else: st.session_state.favorite_projects_selection = [] # Clear if no options
                    else: st.session_state.favorite_projects_selection = []
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
    # Calculate Average Price for Display (using display names)
    avg_price_cols = [f"price {yr}" for yr in st.session_state.available_years_in_data if f"price {yr}" in project_df_display.columns]
    project_df_display['Average Price'] = project_df_display[avg_price_cols].mean(axis=1, skipna=True).fillna(0.0) if avg_price_cols else 0.0

    with st.expander("View Project Details"):
        if not project_df_display.empty:
            cols_for_proj_details = ['project name', 'project type', 'Average Price']
            if 'Project Link' in project_df_display.columns: cols_for_proj_details.append('Project Link')
            col_config_proj_details = {"Average Price": st.column_config.NumberColumn("Avg Price (€/t)", format="€%.2f")}
            if 'Project Link' in project_df_display.columns: col_config_proj_details["Project Link"] = st.column_config.LinkColumn("Project Link", display_text="Visit ->")
            st.dataframe(project_df_display[cols_for_proj_details], column_config=col_config_proj_details, hide_index=True, use_container_width=True)
        else: st.write("No project data to display.")
    st.markdown("---")

if st.session_state.get('data_loaded_successfully', False):
    if not st.session_state.get('selected_projects'): st.warning("⚠️ Please select projects in the sidebar (Section 3).")
    elif not st.session_state.get('selected_years'): st.warning("⚠️ No valid years for planning. Adjust 'Years to Plan' (Section 2).")
    else:
        # Check all required keys for allocation are present and properly initialized
        required_keys_main = ['working_data_full', 'selected_projects', 'selected_years', 'actual_start_year', 'actual_end_year', 'constraint_type', 'annual_targets', 'removal_target_end_year', 'transition_speed', 'category_split', 'favorite_projects_selection', 'min_fulfillment_perc', 'min_alloc_chunk']
        keys_present_main = all(k in st.session_state for k in required_keys_main) and \
                            all(st.session_state.get(k) is not None for k in ['working_data_full', 'selected_projects', 'selected_years', 'actual_start_year', 'actual_end_year', 'constraint_type', 'removal_target_end_year', 'transition_speed', 'category_split', 'min_fulfillment_perc', 'min_alloc_chunk']) and \
                            isinstance(st.session_state.get('annual_targets', {}), dict) and \
                            isinstance(st.session_state.get('favorite_projects_selection', []), list)


        if keys_present_main:
            try:
                fav_proj_main = st.session_state.favorite_projects_selection[0] if st.session_state.favorite_projects_selection else None
                annual_targets_main = st.session_state.annual_targets
                if not annual_targets_main and st.session_state.selected_years:
                    st.warning("Annual targets not set. Using 0 for all years. Configure in Sidebar > Annual Target Settings.")
                    annual_targets_main = {yr: 0 for yr in st.session_state.selected_years}

                if st.session_state.constraint_type == 'Budget': st.info(f"**Budget Mode:** Projects receive budget based on weighted priority & price. May get 0 volume if budget is less than cost of {st.session_state.min_alloc_chunk}t. Adjustment step may add volume later.")
                st.success(f"**Allocation Goal:** Attempting ≥ **{st.session_state.min_fulfillment_perc}%** of annual target {st.session_state.constraint_type}.")
                
                with st.spinner("Calculating portfolio allocation... Please wait."):
                    results_main, summary_df_main = allocate_portfolio(
                        project_data=st.session_state.working_data_full, selected_project_names=st.session_state.selected_projects,
                        selected_years=st.session_state.selected_years, start_year_portfolio=st.session_state.actual_start_year,
                        end_year_portfolio=st.session_state.actual_end_year, constraint_type=st.session_state.constraint_type,
                        annual_targets=annual_targets_main, removal_target_percent_end_year=st.session_state.removal_target_end_year,
                        transition_speed=st.session_state.transition_speed, category_split=st.session_state.category_split,
                        favorite_project=fav_proj_main, min_target_fulfillment_percent=st.session_state.min_fulfillment_perc / 100.0,
                        min_allocation_chunk=st.session_state.min_alloc_chunk)
                
                details_list_main = []
                if results_main:
                    for year_res_main, projects_res_main in results_main.items():
                        if projects_res_main:
                            for proj_res_main in projects_res_main:
                                if isinstance(proj_res_main, dict) and (proj_res_main.get('allocated_volume', 0) >= st.session_state.min_alloc_chunk or proj_res_main.get('allocated_cost', 0) > 1e-6):
                                    details_list_main.append({'year': year_res_main, 'project name': proj_res_main.get('project name'), 'type': proj_res_main.get('type'), 'volume': proj_res_main.get('allocated_volume', 0), 'price': proj_res_main.get('price_used', None), 'cost': proj_res_main.get('allocated_cost', 0.0)})
                
                details_df_main = pd.DataFrame(details_list_main)
                total_portfolio_margin_main = 0.0 # Initialize
                if not details_df_main.empty:
                    details_df_with_margins_main = add_margins_to_details_df(details_df_main.copy(), st.session_state.working_data_full)
                    total_portfolio_margin_main = details_df_with_margins_main['margin'].sum()
                    if not summary_df_main.empty:
                        yearly_margins_main = details_df_with_margins_main.groupby('year')['margin'].sum().reset_index().rename(columns={'margin': 'Total Yearly Margin', 'year': 'Year'})
                        summary_df_main = pd.merge(summary_df_main, yearly_margins_main, on='Year', how='left').fillna({'Total Yearly Margin': 0.0})
                else:
                    details_df_with_margins_main = details_df_main.copy(); details_df_with_margins_main['margin'] = 0.0 # Ensure column exists
                    if not summary_df_main.empty: summary_df_main['Total Yearly Margin'] = 0.0


                st.markdown("## Portfolio Summary"); col_l_main, col_m_main, col_r_main = st.columns([1.5, 1.5, 1.2], gap="large")
                with col_l_main:
                    st.markdown("#### Key Metrics (Overall)")
                    total_cost_all_years, total_volume_all_years, overall_avg_price_main = 0.0, 0, 0.0 # Initialize
                    if not summary_df_main.empty:
                        total_cost_all_years = summary_df_main['Allocated Cost'].sum(); total_volume_all_years = summary_df_main['Allocated Volume'].sum()
                        overall_avg_price_main = total_cost_all_years / total_volume_all_years if total_volume_all_years > 0 else 0.0
                    st.markdown(f"""<div class="metric-box"><b>Total Portfolio Cost</b> € {total_cost_all_years:,.2f}</div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div class="metric-box"><b>Total Portfolio Volume</b> {total_volume_all_years:,.0f} t</div>""", unsafe_allow_html=True)
                with col_m_main:
                    st.markdown("#### &nbsp;")
                    st.markdown(f"""<div class="metric-box"><b>Total Portfolio Margin</b> € {total_portfolio_margin_main:,.2f}</div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div class="metric-box"><b>Overall Average Price</b> € {overall_avg_price_main:,.2f} /t</div>""", unsafe_allow_html=True)
                with col_r_main:
                    st.markdown("#### Volume by Project Type")
                    if not details_df_with_margins_main.empty and details_df_with_margins_main['volume'].sum() > 1e-6:
                        pie_data_main = details_df_with_margins_main.groupby('type')['volume'].sum().reset_index()
                        pie_data_main = pie_data_main[pie_data_main['volume'] > 1e-6]
                        if not pie_data_main.empty:
                            fig_pie_main = px.pie(pie_data_main, values='volume', names='type', color='type', color_discrete_map=type_color_map, height=350)
                            fig_pie_main.update_layout(legend_title_text='Project Type', legend_orientation="h", legend_yanchor="bottom", legend_y=-0.2, legend_xanchor="center", legend_x=0.5, margin=dict(t=5, b=50, l=0, r=0))
                            fig_pie_main.update_traces(textposition='inside', textinfo='percent', sort=False, hole=.3, marker=dict(line=dict(color='#FFFFFF', width=1)))
                            st.plotly_chart(fig_pie_main, use_container_width=True)
                        else: st.caption("No significant volume for pie chart.")
                    else: st.caption("No allocation details for pie chart.")
                
                st.markdown("---")
                if details_df_with_margins_main.empty: st.warning("No detailed project allocations to display further plots or tables.")
                else:
                    st.markdown("### Portfolio Composition & Price Over Time")
                    details_df_with_margins_main['year'] = details_df_with_margins_main['year'].astype(int)
                    summary_plot_data_main = details_df_with_margins_main.groupby(['year', 'type']).agg(volume=('volume', 'sum'), cost=('cost', 'sum'), margin=('margin', 'sum')).reset_index()
                    price_summary_data_main = summary_df_main[['Year', 'Avg. Price']].rename(columns={'Year':'year', 'Avg. Price':'avg_price'}) if not summary_df_main.empty else pd.DataFrame(columns=['year', 'avg_price'])
                    
                    fig_comp_main = make_subplots(specs=[[{"secondary_y": True}]])
                    y_metric_main = 'volume' if st.session_state.constraint_type == 'Volume' else 'cost'
                    y_label_main = f"Allocated {y_metric_main.capitalize()} ({'t' if y_metric_main == 'volume' else '€'})"
                    y_hover_main = f"{y_metric_main.capitalize()}: %{{y:{'{:,.0f}' if y_metric_main == 'volume' else '€{:,.2f}'}}}<extra></extra>"

                    for t_name_main in ['reduction', 'natural removal', 'technical removal']:
                        if t_name_main in summary_plot_data_main['type'].unique():
                            df_type_main = summary_plot_data_main[summary_plot_data_main['type'] == t_name_main]
                            if not df_type_main.empty and df_type_main[y_metric_main].sum() > 1e-6 :
                                fig_comp_main.add_trace(go.Bar(x=df_type_main['year'], y=df_type_main[y_metric_main], name=t_name_main.replace('_', ' ').capitalize(), marker_color=type_color_map.get(t_name_main, default_color), hovertemplate=f'Year: %{{x}}<br>Type: {t_name_main.replace("_", " ").capitalize()}<br>{y_hover_main}'), secondary_y=False)
                    
                    if not price_summary_data_main.empty: fig_comp_main.add_trace(go.Scatter(x=price_summary_data_main['year'], y=price_summary_data_main['avg_price'], name='Avg Price (€/t)', mode='lines+markers', marker_symbol='circle', marker_size=8, line=dict(color='#1B5E20', width=3), hovertemplate='Year: %{x}<br>Avg Price: €%{y:,.2f}/t<extra></extra>'), secondary_y=True)
                    if not summary_df_main.empty and 'Actual Removal Vol %' in summary_df_main.columns: fig_comp_main.add_trace(go.Scatter(x=summary_df_main['Year'], y=summary_df_main['Actual Removal Vol %'], name='Actual Removal Vol %', mode='lines+markers', line=dict(color='darkorange', dash='dash'), marker_symbol='star', marker_size=8, hovertemplate='Year: %{x}<br>Actual Removal: %{y:.1f}%<extra></extra>'), secondary_y=True)
                    
                    y2_max_val_main = 105.0
                    if not price_summary_data_main.empty and 'avg_price' in price_summary_data_main.columns and not price_summary_data_main['avg_price'].empty: y2_max_val_main = max(y2_max_val_main, price_summary_data_main['avg_price'].max() * 1.1 if pd.notna(price_summary_data_main['avg_price'].max()) else y2_max_val_main)
                    if not summary_df_main.empty and 'Actual Removal Vol %' in summary_df_main.columns and not summary_df_main['Actual Removal Vol %'].empty: y2_max_val_main = max(y2_max_val_main, summary_df_main['Actual Removal Vol %'].max() * 1.1 if pd.notna(summary_df_main['Actual Removal Vol %'].max()) else y2_max_val_main)
                    
                    fig_comp_main.update_layout(xaxis_title='Year', yaxis_title=y_label_main, yaxis2_title='Avg Price (€/t) / Actual Removal %', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), barmode='stack', template="plotly_white", margin=dict(t=20, l=0, r=0, b=0), yaxis_rangemode='tozero', yaxis2=dict(rangemode='tozero', range=[0, y2_max_val_main]), hovermode="x unified")
                    if st.session_state.selected_years: fig_comp_main.update_xaxes(tickmode='array', tickvals=st.session_state.selected_years, dtick=1)
                    st.plotly_chart(fig_comp_main, use_container_width=True)

                st.markdown("### Detailed Allocation by Project and Year")
                pivot_display_main = pd.DataFrame()
                if not details_df_with_margins_main.empty:
                    try:
                        details_df_with_margins_main['year'] = pd.to_numeric(details_df_with_margins_main['year'])
                        pivot_intermediate_main = pd.pivot_table(details_df_with_margins_main, values=['volume', 'cost', 'price', 'margin'], index=['project name', 'type'], columns='year', aggfunc={'volume': 'sum', 'cost': 'sum', 'price': 'first', 'margin': 'sum'})
                        
                        if not pivot_intermediate_main.empty:
                            pivot_final_main = pivot_intermediate_main.swaplevel(0, 1, axis=1)
                            metric_order_main = ['volume', 'cost', 'price', 'margin']
                            years_in_pivot_cols_main = sorted([yr for yr in pivot_final_main.columns.get_level_values(0).unique() if isinstance(yr, (int, float, np.number))])
                            
                            if years_in_pivot_cols_main:
                                final_multi_index_main = pd.MultiIndex.from_product([years_in_pivot_cols_main, metric_order_main], names=['year', 'metric'])
                                pivot_final_main = pivot_final_main.reindex(columns=final_multi_index_main).sort_index(axis=1, level=[0, 1])
                            pivot_final_main.index.names = ['Project Name', 'Type']
                            pivot_display_main = pivot_final_main.copy()

                        # Add Total Portfolio Row
                        if not summary_df_main.empty and years_in_pivot_cols_main: # Ensure years_in_pivot_cols_main is defined
                            total_data_dict_main = {}
                            summary_indexed_main = summary_df_main.set_index('Year')
                            for year_total_main in years_in_pivot_cols_main:
                                if year_total_main in summary_indexed_main.index:
                                    s_row = summary_indexed_main.loc[year_total_main]
                                    total_data_dict_main[(year_total_main, 'volume')] = s_row['Allocated Volume']
                                    total_data_dict_main[(year_total_main, 'cost')] = s_row['Allocated Cost']
                                    total_data_dict_main[(year_total_main, 'price')] = s_row['Avg. Price']
                                    total_data_dict_main[(year_total_main, 'margin')] = s_row['Total Yearly Margin']
                            total_row_df_main = pd.DataFrame(total_data_dict_main, index=pd.MultiIndex.from_tuples([('Total Portfolio', 'All Types')], names=['Project Name', 'Type']))
                            pivot_display_main = pd.concat([pivot_display_main, total_row_df_main]) if not pivot_display_main.empty else total_row_df_main
                        
                        if not pivot_display_main.empty:
                            if any(col[1] == 'margin' for col in pivot_display_main.columns if isinstance(col, tuple)):
                                pivot_display_main['Total Margin'] = pivot_display_main.xs('margin', axis=1, level='metric').sum(axis=1)
                            else: pivot_display_main['Total Margin'] = 0.0
                            pivot_display_main = pivot_display_main.fillna(0)
                            
                            formatter_main = {}
                            for col_tuple_main in pivot_display_main.columns:
                                if isinstance(col_tuple_main, tuple) and len(col_tuple_main) == 2:
                                    metric_val = col_tuple_main[1]
                                    if metric_val == 'volume': formatter_main[col_tuple_main] = '{:,.0f} t'
                                    elif metric_val == 'cost': formatter_main[col_tuple_main] = '€{:,.2f}'
                                    elif metric_val == 'price': formatter_main[col_tuple_main] = lambda x_val: f'€{x_val:,.2f}/t' if pd.notna(x_val) and x_val != 0 else '-'
                                    elif metric_val == 'margin': formatter_main[col_tuple_main] = '€{:,.2f}'
                                elif col_tuple_main == 'Total Margin': formatter_main[col_tuple_main] = '€{:,.2f}'
                            st.dataframe(pivot_display_main.style.format(formatter_main, na_rep="-"), use_container_width=True)
                        else: st.info("No data for detailed allocation table.")
                    except Exception as e_pivot: st.error(f"Could not create detailed allocation table: {e_pivot}"); st.error(f"Traceback: {traceback.format_exc()}")
                else: st.info("No allocation details available to generate the detailed table.")
                
                if not pivot_display_main.empty:
                    csv_df_main = pivot_display_main.copy()
                    csv_df_main.columns = [f"{str(col[0])}_{col[1]}" if isinstance(col, tuple) else str(col) for col in csv_df_main.columns.values]
                    csv_df_main = csv_df_main.reset_index()
                    try:
                        csv_string_main = csv_df_main.to_csv(index=False).encode('utf-8')
                        st.markdown("---"); st.download_button(label="Download Detailed Allocation (CSV)", data=csv_string_main, file_name=f"portfolio_allocation_{datetime.date.today()}.csv", mime='text/csv', key='download-csv')
                    except Exception as e_csv_main: st.error(f"Error generating CSV: {e_csv_main}")

            except ValueError as e_val_main: st.error(f"Config/Allocation Error: {e_val_main}")
            except KeyError as e_key_main: st.error(f"Data Error (Missing Key): '{e_key_main}'. Check CSV & selections. Trace: {traceback.format_exc()}")
            except Exception as e_gen_main: st.error(f"Unexpected error in main processing: {e_gen_main}"); st.error(f"Traceback: {traceback.format_exc()}")
        else: st.error("⚠️ Missing required settings for allocation. Please ensure data is loaded and all sidebar settings (especially planning horizon and project selections) are correctly configured.")

# --- Footer ---
st.markdown("---")
try:
    zurich_tz = pytz.timezone('Europe/Zurich')
    now_zurich = datetime.datetime.now(zurich_tz)
    st.caption(f"Report generated: {now_zurich.strftime('%Y-%m-%d %H:%M:%S %Z')}")
except Exception:
    now_local = datetime.datetime.now()
    st.caption(f"Report generated: {now_local.strftime('%Y-%m-%d %H:%M:%S')} (Timezone: Server Local)")
