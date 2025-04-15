# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import copy
import datetime
import pytz # Added for timezone handling

# ==================================
# Configuration & Theming (Green Sidebar, White Main)
# ==================================
st.set_page_config(layout="wide")
# --- CSS ---
css = """
<style>
    /* Main App background */
    .stApp { background-color: #FFFFFF; } /* White */

    /* Sidebar (Green Theme) */
    [data-testid="stSidebar"] {
        background-color: #C8E6C9; /* Light Green */
        padding-top: 2rem;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] .stText {
        color: #1B5E20; /* Dark Green Text */
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .st-bq {
        color: #1B5E20 !important; /* Dark Green Text */
    }
    [data-testid="stSidebar"] .stButton>button {
        background-color: #4CAF50; color: white; border: 1px solid #388E3C;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #388E3C; color: white; border: 1px solid #1B5E20;
    }
    /* Widgets in Sidebar */
    [data-testid="stSidebar"] .stNumberInput input,
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"],
    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"],
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stDateInput input,
    [data-testid="stSidebar"] .stTimeInput input {
        border: 1px solid #A5D6A7 !important; /* Light green border */
        background-color: #FFFFFF; /* White background for inputs */
        color: #1B5E20; /* Dark green text inside input */
    }
    /* Slider specific styling (Neutral Colors) */
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] {
        background: #EEEEEE; /* Light Grey base track */
    }
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div:nth-child(1) {
         background-color: #DDDDDD !important; /* Neutral grey for selected range track */
    }
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div:nth-child(3) {
        background-color: #888888 !important; /* Neutral grey handle */
        border-color: #777777 !important;
     }

    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] { background-color: #66BB6A; color: white; }
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] span:first-child { background-color: #A5D6A7 !important; border: 2px solid #388E3C !important; } /* Light green circle */
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] span:first-child[aria-checked="true"] { background-color: #388E3C !important; } /* Dark green selected */

    /* Main content Margins */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; padding-left: 3rem; padding-right: 3rem; max-width: 100%; }

    /* Main Content Titles & Text */
    h1, h2, h3, h4, h5, h6 { color: #1B5E20; } /* Dark Green */
    .block-container, .block-container p, .block-container li, .block-container div:not(.stPlotlyChart) { color: #1B5E20; }
    .stMarkdown p, .stText { color: #1B5E20; }
    .stAlert p { color: inherit !important; }
    .stException p { color: inherit !important; }

    /* Dataframes */
    .stDataFrame { border: 1px solid #A5D6A7; }

    /* --- Key Metrics Styling (Individual Boxes) --- */
    .key-metrics-line {
        /* Container just provides margin below */
        margin-bottom: 25px;
    }
    /* Style EACH metric box within the container/columns */
    .key-metrics-line .stMetric {
        background-color: #C8E6C9 !important; /* Light Green background */
        border: 2px solid #1B5E20 !important; /* Dark Green border */
        border-radius: 8px !important;         /* Rounded edges */
        padding: 10px 15px !important;         /* Padding inside */
        text-align: center;
        color: #1B5E20 !important; /* Dark Green text */
    }
     /* Metric Label Style */
     .key-metrics-line .stMetric > label {
        color: #1B5E20 !important; /* Dark Green text */
        font-weight: bold;
        margin-bottom: 5px !important;
     }
     /* Metric Value Style */
     .key-metrics-line .stMetric > div > div {
         color: #1B5E20 !important; /* Dark Green text */
         font-size: 1.4em; /* Adjust size */
         line-height: 1.2;
         margin-top: 0px !important;
     }
    /* Style for Horizon metric value */
     .key-metrics-line .stMetric[aria-label="Horizon"] > div > div {
        font-size: 1.2em !important; /* Slightly smaller for text range */
        font-weight: bold;
     }
    /* Delta styling (optional) */
     .key-metrics-line .stMetric > div > div:nth-child(2) {
        font-size: 0.85rem;
        color: #388E3C !important; /* Medium green for delta */
     }
     /* Style for the pie chart column */
     .key-metrics-line [data-testid="stVerticalBlock"]:last-child {
         text-align: center;
         /* Add padding top to align pie chart better if needed */
         padding-top: 5px;
     }

</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ==================================
# Data Loading Function
# ==================================
@st.cache_data
def load_and_prepare_data(uploaded_file):
    # (Function unchanged)
    """Loads, standardizes, validates, and prepares the uploaded project data."""
    try:
        data = pd.read_csv(uploaded_file)
        data.columns = data.columns.str.lower().str.strip().str.replace(' ', '_')
        core_cols_std_internal = ['project_name', 'project_type', 'priority']
        missing_essential = [col for col in core_cols_std_internal if col not in data.columns]
        if missing_essential: return None, f"Missing essential columns (expected: {', '.join(core_cols_std_internal)}). Found: {', '.join(data.columns)}", [], [], []
        numeric_prefixes_std_internal = ['price_', 'available_volume_']
        cols_to_convert = ['priority']; available_years = set(); price_cols_found_internal = []; volume_cols_found_internal = []
        for col in data.columns:
            for prefix in numeric_prefixes_std_internal:
                if col.startswith(prefix):
                    year_part = col[len(prefix):]
                    if year_part.isdigit(): cols_to_convert.append(col); available_years.add(int(year_part)); (price_cols_found_internal if prefix == 'price_' else volume_cols_found_internal).append(col); break
        for col in list(set(cols_to_convert)):
            if col in data.columns: data[col] = pd.to_numeric(data[col], errors='coerce')
        data['priority'] = data['priority'].fillna(0)
        for col in volume_cols_found_internal:
            if col in data.columns: data[col] = data[col].fillna(0).apply(lambda x: max(0, int(x)) if pd.notna(x) else 0).clip(lower=0).astype(int)
        for col in price_cols_found_internal:
            if col in data.columns: data[col] = data[col].fillna(0.0).apply(lambda x: max(0.0, float(x)) if pd.notna(x) else 0.0).clip(lower=0.0).astype(float)
        available_years = sorted(list(available_years))
        if not available_years: return None, "No valid year data columns found (e.g., 'price_YYYY', 'available_volume_YYYY').", [], [], []
        invalid_types_found = []
        if 'project_type' in data.columns:
            data['project_type'] = data['project_type'].astype(str).str.lower().str.strip(); valid_types = ['reduction', 'technical removal', 'natural removal']; invalid_types_df = data[~data['project_type'].isin(valid_types)]
            if not invalid_types_df.empty: invalid_types_found = invalid_types_df['project_type'].unique().tolist(); data = data[data['project_type'].isin(valid_types)].copy()
        else: return None, "Missing 'project_type' column.", available_years, [], []
        column_mapping_to_display = {col: col.replace('_', ' ') for col in data.columns}; data.rename(columns=column_mapping_to_display, inplace=True)
        project_names_list = sorted(data['project name'].unique().tolist()) if 'project name' in data.columns else []
        return data, None, available_years, project_names_list, invalid_types_found
    except Exception as e: return None, f"Error reading or processing CSV: {e}", [], [], []

# ==================================
# Allocation Function
# ==================================
@st.cache_data
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
    priority_boost_percent: int = 10
) -> tuple[dict, pd.DataFrame]:
    # (Function unchanged from previous version)
    """
    Allocates portfolio based on user selection and transition goals.
    Builds summary DataFrame column by column to avoid duplicates.
    Handles years within selected_years where data columns might be missing.
    """
    portfolio_details = {year: {} for year in selected_years}
    years_list = []; target_values_list = []; allocated_volumes_list = []; shortfall_list = []
    allocated_costs_list = []; avg_prices_list = []; actual_removal_pct_list = []; target_removal_pct_list = []
    summary_col_suffix = "Volume" if constraint_type == "Volume" else "Budget"
    selected_project_names_set = frozenset(selected_project_names)

    if not selected_project_names_set: return {}, pd.DataFrame()
    project_data_selected = project_data[project_data['project name'].isin(selected_project_names)].copy()
    if project_data_selected.empty: return {}, pd.DataFrame()
    all_project_types_in_selection = project_data_selected['project type'].unique()
    if len(all_project_types_in_selection) == 0: st.warning("Selected projects have no valid project types."); return {}, pd.DataFrame()
    is_reduction_selected = 'reduction' in all_project_types_in_selection
    if not isinstance(start_year_portfolio, (int, float)) or not isinstance(end_year_portfolio, (int, float)) or end_year_portfolio < start_year_portfolio: total_years_duration = 0
    else: total_years_duration = end_year_portfolio - start_year_portfolio
    num_cols_to_check = ['priority']
    for yr in selected_years: num_cols_to_check.extend([f"price {yr}", f"available volume {yr}"])
    for col in set(num_cols_to_check):
        if col in project_data_selected.columns:
            current_dtype = project_data_selected[col].dtype
            if col == 'priority' and not pd.api.types.is_numeric_dtype(current_dtype): project_data_selected[col] = pd.to_numeric(project_data_selected[col], errors='coerce').fillna(0)
            elif col.startswith("available volume") and not pd.api.types.is_integer_dtype(current_dtype): project_data_selected[col] = pd.to_numeric(project_data_selected[col], errors='coerce').fillna(0).apply(lambda x: int(x) if pd.notna(x) else 0).clip(lower=0)
            elif col.startswith("price") and not pd.api.types.is_float_dtype(current_dtype): project_data_selected[col] = pd.to_numeric(project_data_selected[col], errors='coerce').fillna(0.0).apply(lambda x: float(x) if pd.notna(x) else 0.0).clip(lower=0.0)

    for year in selected_years:
        yearly_target = annual_targets.get(year, 0)
        price_col = f"price {year}"; volume_col = f"available volume {year}"
        allocation_dict_for_year = {}; fractional_allocations = {}; calculated_target_removal_pct = 0.0
        if price_col not in project_data_selected.columns or volume_col not in project_data_selected.columns:
            years_list.append(year); target_values_list.append(yearly_target); allocated_volumes_list.append(0); allocated_costs_list.append(0); avg_prices_list.append(0); shortfall_skipped = yearly_target if constraint_type == 'Budget' else int(round(yearly_target)); shortfall_list.append(max(0, shortfall_skipped)); actual_removal_pct_list.append(0); target_removal_pct_list.append(0); portfolio_details[year] = []
            continue
        usable_data_exists = not project_data_selected[project_data_selected[price_col].notna() & (project_data_selected[price_col] > 0) & project_data_selected[volume_col].notna() & (project_data_selected[volume_col] >= 0)].empty
        if not usable_data_exists:
             years_list.append(year); target_values_list.append(yearly_target); allocated_volumes_list.append(0); allocated_costs_list.append(0); avg_prices_list.append(0); shortfall_skipped = yearly_target if constraint_type == 'Budget' else int(round(yearly_target)); shortfall_list.append(max(0, shortfall_skipped)); actual_removal_pct_list.append(0); target_removal_pct_list.append(0); portfolio_details[year] = []
             continue
        if yearly_target <= 0:
            years_list.append(year); target_values_list.append(0); allocated_volumes_list.append(0); allocated_costs_list.append(0); avg_prices_list.append(0); shortfall_list.append(0); actual_removal_pct_list.append(0); target_removal_pct_list.append(0); portfolio_details[year] = []
            continue
        # Calculate Target % Mix (unchanged)
        target_percentages = {} # ... (logic as before) ...
        if is_reduction_selected:
            start_removal_pct = 0.10; end_removal_pct = removal_target_percent_end_year
            if total_years_duration <= 0: progress = 1.0
            else: progress = max(0, min(1, (year - start_year_portfolio) / total_years_duration))
            exponent = 0.1 + (11 - transition_speed) * 0.2; progress_factor = progress ** exponent
            target_removal_pct_year = start_removal_pct + (end_removal_pct - start_removal_pct) * progress_factor
            min_target_pct = min(start_removal_pct, end_removal_pct); max_target_pct = max(start_removal_pct, end_removal_pct)
            target_removal_pct_year = max(min_target_pct, min(max_target_pct, target_removal_pct_year))
            target_tech_pct_year = target_removal_pct_year * category_split.get('technical removal', 0); target_nat_pct_year = target_removal_pct_year * category_split.get('natural removal', 0); target_red_pct_year = max(0.0, 1.0 - target_tech_pct_year - target_nat_pct_year); target_percentages = {'reduction': target_red_pct_year, 'technical removal': target_tech_pct_year, 'natural removal': target_nat_pct_year}
        else:
            tech_share = category_split.get('technical removal', 0); nat_share = category_split.get('natural removal', 0); total_removal_share = tech_share + nat_share
            tech_selected = 'technical removal' in all_project_types_in_selection; nat_selected = 'natural removal' in all_project_types_in_selection
            if total_removal_share > 1e-9:
                tech_alloc = tech_share / total_removal_share if tech_selected else 0; nat_alloc = nat_share / total_removal_share if nat_selected else 0; total_alloc = tech_alloc + nat_alloc
                if total_alloc > 1e-9: target_percentages['technical removal'] = tech_alloc / total_alloc if tech_selected else 0; target_percentages['natural removal'] = nat_alloc / total_alloc if nat_selected else 0
                else:
                    if tech_selected and nat_selected: target_percentages = {'technical removal': 0.5, 'natural removal': 0.5}
                    elif tech_selected: target_percentages['technical removal'] = 1.0
                    elif nat_selected: target_percentages['natural removal'] = 1.0
            else:
                num_removal_types_selected = (1 if tech_selected else 0) + (1 if nat_selected else 0)
                if num_removal_types_selected > 0: equal_share = 1.0 / num_removal_types_selected;
                if tech_selected: target_percentages['technical removal'] = equal_share
                if nat_selected: target_percentages['natural removal'] = equal_share
            target_percentages['reduction'] = 0.0
        current_sum = sum(target_percentages.values()) # Normalize
        if abs(current_sum - 1.0) > 1e-6 and current_sum > 0: norm_factor = 1.0 / current_sum; target_percentages = {pt: share * norm_factor for pt, share in target_percentages.items()}; target_percentages = {pt: share if share > 1e-9 else 0.0 for pt, share in target_percentages.items()}; current_sum = sum(target_percentages.values());
        if abs(current_sum - 1.0) > 1e-6 and current_sum > 0: norm_factor = 1.0 / current_sum; target_percentages = {pt: share * norm_factor for pt, share in target_percentages.items()}
        calculated_target_removal_pct = (target_percentages.get('technical removal', 0) + target_percentages.get('natural removal', 0)) * 100

        # Initial Allocation (Pass 1) (unchanged logic)
        year_initial_total_allocated_vol = 0; year_initial_total_allocated_cost = 0
        for idx, proj_row_init in project_data_selected.iterrows():
             proj_name_init = proj_row_init['project name']
             if proj_name_init not in allocation_dict_for_year: price_init = proj_row_init.get(price_col, 0); vol_init = proj_row_init.get(volume_col, 0); allocation_dict_for_year[proj_name_init] = {'project name': proj_name_init, 'type': proj_row_init['project type'], 'allocated_volume': 0, 'allocated_cost': 0, 'price_used': price_init if pd.notna(price_init) else 0, 'priority_applied': 0, 'initial_available': vol_init if pd.notna(vol_init) else 0, 'remaining_available': vol_init if pd.notna(vol_init) else 0}
        for project_type in target_percentages:
            type_share = target_percentages.get(project_type, 0);
            if type_share <= 0: continue
            target_resource_for_type = yearly_target * type_share
            projects_in_type = project_data_selected[(project_data_selected['project type'] == project_type) & (project_data_selected[price_col].notna()) & (project_data_selected[price_col] > 0) & (project_data_selected[volume_col].notna()) & (project_data_selected[volume_col] > 0)].copy()
            if projects_in_type.empty: continue
            total_priority_in_type = projects_in_type['priority'].sum()
            if total_priority_in_type <= 1e-9: num_projects = len(projects_in_type); equal_prio = 1.0 / num_projects if num_projects > 0 else 0; projects_in_type['norm_prio_base'] = equal_prio
            else: projects_in_type['norm_prio_base'] = projects_in_type['priority'] / total_priority_in_type
            current_norm_priorities = projects_in_type.set_index('project name')['norm_prio_base'].to_dict(); final_adjusted_priorities = current_norm_priorities.copy()
            total_boosted_prio = None
            if favorite_project and favorite_project in final_adjusted_priorities:
                boost_factor = 1.0 + priority_boost_percent / 100.0; boosted_prios = {p: prio for p, prio in final_adjusted_priorities.items()}
                if favorite_project in boosted_prios: boosted_prios[favorite_project] *= boost_factor; total_boosted_prio = sum(boosted_prios.values());
                if total_boosted_prio is not None and total_boosted_prio > 1e-9: final_adjusted_priorities = {p: prio / total_boosted_prio for p, prio in boosted_prios.items()}
            for proj_name_prio, prio_val in final_adjusted_priorities.items():
                 if proj_name_prio in allocation_dict_for_year: allocation_dict_for_year[proj_name_prio]['priority_applied'] = prio_val
            total_weight_for_type = 0; project_weights = {}
            if constraint_type == 'Budget':
                for _, project_row in projects_in_type.iterrows(): proj_name = project_row['project name']; p_prio_final = final_adjusted_priorities.get(proj_name, 0); p_price = project_row[price_col];
                if p_price > 0: current_weight = p_prio_final / p_price; project_weights[proj_name] = current_weight; total_weight_for_type += current_weight
            for _, project_row in projects_in_type.iterrows():
                proj_name = project_row['project name']; proj_prio_final = final_adjusted_priorities.get(proj_name, 0)
                if pd.isna(project_row[volume_col]) or pd.isna(project_row[price_col]): continue
                available_vol = project_row[volume_col]; price = project_row[price_col]; allocated_volume_frac = 0.0; allocated_volume_int = 0; allocated_cost = 0
                if price <= 0 or (proj_prio_final <= 0 and total_priority_in_type > 1e-9): continue
                if constraint_type == 'Volume': project_target_volume_frac = target_resource_for_type * proj_prio_final; allocated_volume_frac = min(project_target_volume_frac, available_vol); fractional_allocations[proj_name] = allocated_volume_frac
                elif constraint_type == 'Budget': project_target_volume_frac = 0;
                if total_weight_for_type > 1e-9: normalized_weight = project_weights.get(proj_name, 0) / total_weight_for_type; project_target_budget = target_resource_for_type * normalized_weight; project_target_volume_frac = project_target_budget / price if price > 0 else 0; allocated_volume_frac = min(project_target_volume_frac, available_vol)
                allocated_volume_int = int(max(0, math.floor(allocated_volume_frac)))
                if allocated_volume_int > 0 and allocated_volume_int <= available_vol :
                    allocated_cost = allocated_volume_int * price; allocation_dict_for_year[proj_name].update({'allocated_volume': allocated_volume_int, 'allocated_cost': allocated_cost, 'remaining_available': available_vol - allocated_volume_int});
                    year_initial_total_allocated_vol += allocated_volume_int; year_initial_total_allocated_cost += allocated_cost

        # Remainder Distribution (Pass 2) (unchanged logic)
        year_final_total_allocated_vol = year_initial_total_allocated_vol; year_final_total_allocated_cost = year_initial_total_allocated_cost
        if constraint_type == 'Volume':
            volume_shortfall_pass1 = int(round(yearly_target - year_initial_total_allocated_vol))
            if volume_shortfall_pass1 > 0:
                eligible_projects_remainder = [] # ... (remainder logic) ...
                for proj_name, details in allocation_dict_for_year.items():
                     original_row = project_data_selected[ (project_data_selected['project name'] == proj_name) ]
                     if not original_row.empty and details['remaining_available'] > 0 and pd.notna(original_row.iloc[0][volume_col]) and original_row.iloc[0][volume_col] > 0 and pd.notna(original_row.iloc[0][price_col]) and original_row.iloc[0][price_col] > 0: fractional_part_lost = fractional_allocations.get(proj_name, 0.0) - details['allocated_volume']; eligible_projects_remainder.append({'name': proj_name, 'fraction_lost': fractional_part_lost, 'priority': details['priority_applied'], 'remaining_capacity': details['remaining_available'], 'price': details['price_used']})
                eligible_projects_remainder.sort(key=lambda x: (-x['fraction_lost'], -x['priority'])); allocated_remainder_vol = 0
                for project_info in eligible_projects_remainder:
                    if allocated_remainder_vol >= volume_shortfall_pass1: break
                    proj_name_rem = project_info['name']; can_allocate_now = min(project_info['remaining_capacity'], volume_shortfall_pass1 - allocated_remainder_vol); units_to_allocate = int(max(0, can_allocate_now))
                    if units_to_allocate > 0: price_rem = allocation_dict_for_year[proj_name_rem]['price_used']; allocation_dict_for_year[proj_name_rem]['allocated_volume'] += units_to_allocate; allocation_dict_for_year[proj_name_rem]['allocated_cost'] += units_to_allocate * price_rem; allocation_dict_for_year[proj_name_rem]['remaining_available'] -= units_to_allocate; year_final_total_allocated_vol += units_to_allocate; year_final_total_allocated_cost += units_to_allocate * price_rem; allocated_remainder_vol += units_to_allocate
        elif constraint_type == 'Budget':
            budget_shortfall_pass1 = yearly_target - year_initial_total_allocated_cost
            while budget_shortfall_pass1 > 1e-6: # ... (remainder logic) ...
                cheapest_project_name_rem = None; cheapest_price_rem = float('inf'); candidate_found = False
                for proj_name, details in allocation_dict_for_year.items():
                     original_row = project_data_selected[ (project_data_selected['project name'] == proj_name) ]
                     if not original_row.empty and details['remaining_available'] > 0 and details['price_used'] > 0:
                             if details['price_used'] < cheapest_price_rem: cheapest_price_rem = details['price_used']; cheapest_project_name_rem = proj_name; candidate_found = True
                             elif details['price_used'] == cheapest_price_rem and details['priority_applied'] > allocation_dict_for_year.get(cheapest_project_name_rem, {}).get('priority_applied', -1): cheapest_project_name_rem = proj_name
                if not candidate_found or cheapest_price_rem > budget_shortfall_pass1 or cheapest_price_rem <= 0: break
                allocation_dict_for_year[cheapest_project_name_rem]['allocated_volume'] += 1; allocation_dict_for_year[cheapest_project_name_rem]['allocated_cost'] += cheapest_price_rem; allocation_dict_for_year[cheapest_project_name_rem]['remaining_available'] -= 1; year_final_total_allocated_vol += 1; year_final_total_allocated_cost += cheapest_price_rem; budget_shortfall_pass1 -= cheapest_price_rem

        # Final Calculations & Append to Lists (unchanged)
        final_allocated_volume_this_year = sum(p['allocated_volume'] for p in allocation_dict_for_year.values()); final_allocated_cost_this_year = sum(p['allocated_cost'] for p in allocation_dict_for_year.values())
        portfolio_details[year] = list(allocation_dict_for_year.values())
        avg_price_this_year = (final_allocated_cost_this_year / final_allocated_volume_this_year) if final_allocated_volume_this_year > 0 else 0
        removal_vol = sum(p['allocated_volume'] for p in portfolio_details[year] if p.get('type') != 'reduction'); actual_removal_pct_this_year = (removal_vol / final_allocated_volume_this_year * 100) if final_allocated_volume_this_year > 0 else 0
        shortfall_this_year = 0
        if constraint_type == 'Volume': shortfall_this_year = max(0, int(round(yearly_target)) - final_allocated_volume_this_year)
        elif constraint_type == 'Budget': final_shortfall_val = yearly_target - final_allocated_cost_this_year; shortfall_this_year = final_shortfall_val if final_shortfall_val > 1e-6 else 0.0
        years_list.append(year); target_values_list.append(yearly_target); allocated_volumes_list.append(final_allocated_volume_this_year); shortfall_list.append(shortfall_this_year); allocated_costs_list.append(final_allocated_cost_this_year); avg_prices_list.append(avg_price_this_year); actual_removal_pct_list.append(actual_removal_pct_this_year); target_removal_pct_list.append(calculated_target_removal_pct)
        # End of Year Loop

    # Consolidate Summary Data (unchanged)
    shortfall_col_name = f'{summary_col_suffix} Shortfall'
    summary_data_dict = {'Year': years_list, 'Target Value': target_values_list, 'Allocated Volume': allocated_volumes_list, shortfall_col_name: shortfall_list, 'Allocated Cost': allocated_costs_list, 'Avg. Price': avg_prices_list, 'Actual Removal Vol %': actual_removal_pct_list, 'Target Removal Vol %': target_removal_pct_list}
    yearly_summary_df = pd.DataFrame(summary_data_dict)
    target_type = int if constraint_type == 'Volume' else float; shortfall_type = int if constraint_type == 'Volume' else float
    if 'Target Value' in yearly_summary_df.columns: yearly_summary_df['Target Value'] = yearly_summary_df['Target Value'].astype(target_type)
    if shortfall_col_name in yearly_summary_df.columns: yearly_summary_df[shortfall_col_name] = yearly_summary_df[shortfall_col_name].astype(shortfall_type)
    if 'Allocated Volume' in yearly_summary_df.columns: yearly_summary_df['Allocated Volume'] = yearly_summary_df['Allocated Volume'].astype(int)
    display_cols_order = ['Year', 'Target Value', 'Allocated Volume', shortfall_col_name, 'Allocated Cost', 'Avg. Price', 'Actual Removal Vol %', 'Target Removal Vol %']
    final_cols_ordered = [col for col in display_cols_order if col in yearly_summary_df.columns]
    final_df_cols = yearly_summary_df.columns
    if final_df_cols.duplicated().any(): print(f"!!! WARNING: Duplicate columns detected before return: {final_df_cols[final_df_cols.duplicated()].tolist()}"); yearly_summary_df = yearly_summary_df.loc[:, ~final_df_cols.duplicated()]
    final_cols_ordered = [col for col in display_cols_order if col in yearly_summary_df.columns]
    if 'Year' in yearly_summary_df.columns: yearly_summary_df['Year'] = yearly_summary_df['Year'].astype(int)

    if yearly_summary_df.empty: return portfolio_details, pd.DataFrame(columns=final_cols_ordered)
    else: return portfolio_details, yearly_summary_df[final_cols_ordered]


# ==================================
# Streamlit App Layout & Logic
# ==================================
# Sidebar Section
with st.sidebar:
    st.markdown("## 1. Load Data")
    df_upload = st.file_uploader("Upload Project Data CSV", type="csv", key="uploader_sidebar")
    # Initialize Session State
    if 'working_data_full' not in st.session_state: st.session_state.working_data_full = None
    if 'selected_years' not in st.session_state: st.session_state.selected_years = []
    if 'selected_projects' not in st.session_state: st.session_state.selected_projects = []
    if 'project_names' not in st.session_state: st.session_state.project_names = []
    if 'favorite_projects_selection' not in st.session_state: st.session_state.favorite_projects_selection = []
    if 'actual_start_year' not in st.session_state: st.session_state.actual_start_year = None
    if 'actual_end_year' not in st.session_state: st.session_state.actual_end_year = None
    if 'available_years_in_data' not in st.session_state: st.session_state.available_years_in_data = []
    if 'constraint_type' not in st.session_state: st.session_state.constraint_type = 'Volume'
    if 'removal_target_end_year' not in st.session_state: st.session_state.removal_target_end_year = 0.8
    if 'transition_speed' not in st.session_state: st.session_state.transition_speed = 5
    if 'category_split' not in st.session_state: st.session_state.category_split = {'technical removal': 0.5, 'natural removal': 0.5}
    if 'annual_targets' not in st.session_state: st.session_state.annual_targets = {}
    if 'data_loaded_successfully' not in st.session_state: st.session_state.data_loaded_successfully = False
    if 'start_year_input' not in st.session_state: st.session_state.start_year_input = None
    if 'end_year_input' not in st.session_state: st.session_state.end_year_input = None
    if 'removal_preference_slider' not in st.session_state: st.session_state.removal_preference_slider = 5

    # Data Loading Logic
    if df_upload:
        try:
            data, error_msg, available_years_in_data, project_names_list, invalid_types_found = load_and_prepare_data(df_upload)
            if invalid_types_found: st.sidebar.warning(f"Invalid project types ignored: {', '.join(invalid_types_found)}")
            if error_msg: st.sidebar.error(error_msg); st.session_state.data_loaded_successfully = False; st.session_state.working_data_full = None; st.session_state.project_names = []; st.session_state.available_years_in_data = []; st.session_state.selected_years = []; st.session_state.annual_targets = {}; st.session_state.start_year_input = None; st.session_state.end_year_input = None;
            else:
                data_just_loaded = not st.session_state.data_loaded_successfully
                st.session_state.project_names = project_names_list; st.session_state.available_years_in_data = available_years_in_data; st.session_state.working_data_full = data; st.session_state.data_loaded_successfully = True
                current_selection = st.session_state.get('selected_projects', []); valid_current_selection = [p for p in current_selection if p in project_names_list]
                if not valid_current_selection and project_names_list: st.session_state.selected_projects = project_names_list
                elif valid_current_selection: st.session_state.selected_projects = valid_current_selection
                else: st.session_state.selected_projects = []
                if data_just_loaded or st.session_state.start_year_input is None:
                    if available_years_in_data:
                         st.session_state.start_year_input = min(available_years_in_data)
                         potential_end = st.session_state.start_year_input + 4
                         st.session_state.end_year_input = min(potential_end, max(available_years_in_data))
                if data_just_loaded: st.sidebar.success("Data loaded successfully!"); st.rerun()
        except Exception as e: st.sidebar.error(f"Error processing file: {e}"); st.session_state.data_loaded_successfully = False; st.session_state.working_data_full = None; st.session_state.project_names = []; st.session_state.available_years_in_data = []; st.session_state.selected_years = []; st.session_state.annual_targets = {}; st.session_state.start_year_input = None; st.session_state.end_year_input = None;

    # Sections 2 & 3
    if st.session_state.get('data_loaded_successfully', False):
        data_df = st.session_state.working_data_full # Get data frame from state
        available_years_in_data = st.session_state.available_years_in_data; project_names_list = st.session_state.project_names
        st.markdown("## 2. Portfolio Settings")
        if not available_years_in_data:
            st.sidebar.warning("No usable year data found.");
            if st.session_state.selected_years: st.session_state.selected_years = []
            if st.session_state.annual_targets: st.session_state.annual_targets = {}
            st.session_state.actual_start_year = None; st.session_state.actual_end_year = None
            st.session_state.start_year_input = None; st.session_state.end_year_input = None;
        else:
            min_year_data = min(available_years_in_data)
            max_year_data = max(available_years_in_data)

            # --- Year Selection using Number Inputs ---
            st.sidebar.markdown("### Planning Horizon")
            start_default = st.session_state.start_year_input if st.session_state.start_year_input is not None else min_year_data
            end_default = st.session_state.end_year_input if st.session_state.end_year_input is not None else min(start_default + 4, max_year_data)

            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_year_input_val = st.number_input("Start Year", min_value=min_year_data, max_value=max_year_data, value=start_default, step=1, format="%d", key="start_year_input_widget")
            with col2:
                end_year_input_val = st.number_input("End Year", min_value=min_year_data, max_value=max_year_data, value=end_default, step=1, format="%d", key="end_year_input_widget")

            # --- Update State & Rerun Logic ---
            input_years_valid = True
            if end_year_input_val < start_year_input_val:
                st.sidebar.error("End Year must be >= Start Year.")
                input_years_valid = False
                temp_selected_years = []
            else:
                temp_selected_years = list(range(start_year_input_val, end_year_input_val + 1))

            start_changed = (st.session_state.start_year_input != start_year_input_val)
            end_changed = (st.session_state.end_year_input != end_year_input_val)
            list_changed = (st.session_state.selected_years != temp_selected_years)

            # Update state variables from widgets
            st.session_state.start_year_input = start_year_input_val
            st.session_state.end_year_input = end_year_input_val

            # Update selected_years list and prune targets if list changed
            if list_changed:
                 st.session_state.selected_years = temp_selected_years
                 if input_years_valid:
                      st.session_state.annual_targets = { yr: val for yr, val in st.session_state.annual_targets.items() if yr in temp_selected_years }

            # Rerun if inputs changed and the range is valid, or if the valid list changed
            if input_years_valid and (start_changed or end_changed or list_changed):
                 st.rerun() # Force refresh ensures consistency

            # Display Planning Horizon & Constraint Type
            if not st.session_state.selected_years or not input_years_valid:
                st.session_state.actual_start_year = None; st.session_state.actual_end_year = None;
                if st.session_state.annual_targets: st.session_state.annual_targets = {}
            else:
                current_start = min(st.session_state.selected_years); current_end = max(st.session_state.selected_years)
                if st.session_state.actual_start_year != current_start: st.session_state.actual_start_year = current_start
                if st.session_state.actual_end_year != current_end: st.session_state.actual_end_year = current_end
                st.sidebar.markdown(f"Selected Range: **{st.session_state.actual_start_year}–{st.session_state.actual_end_year}**")

                st.session_state.constraint_type = st.radio("Constraint Type:", ('Volume', 'Budget'), index=['Volume', 'Budget'].index(st.session_state.get('constraint_type', 'Volume')), key='constraint_type_sidebar', horizontal=True); constraint_type = st.session_state.constraint_type

                # Annual Target Inputs Expander (unchanged)
                with st.expander(f"Set Annual Targets ({constraint_type})", expanded=False):
                    if not st.session_state.selected_years or not input_years_valid: st.caption("Select valid Start/End years above.")
                    else:
                        updated_targets_this_run = {}
                        is_budget = (constraint_type == 'Budget'); default_val = 100000.0 if is_budget else 1000; step_val = 1000.0 if is_budget else 100; format_val = "%.2f" if is_budget else "%d"; label_suffix = "€" if is_budget else "tonnes"; min_val = 0.0 if is_budget else 0
                        for year in st.session_state.selected_years:
                            raw_target_val = st.session_state.annual_targets.get(year, default_val)
                            if is_budget:
                                try: current_target_for_year = float(raw_target_val)
                                except (ValueError, TypeError): current_target_for_year = default_val
                            else:
                                try: current_target_for_year = int(raw_target_val)
                                except (ValueError, TypeError): current_target_for_year = default_val
                            target_input = st.number_input(label=f"Target {label_suffix} for {year}", min_value=min_val, step=step_val, value=current_target_for_year, format=format_val, key=f"target_input_{year}")
                            updated_targets_this_run[year] = target_input
                        st.session_state.annual_targets = updated_targets_this_run

                # Removal Transition Settings (unchanged)
                st.sidebar.markdown("### Removal Volume Transition"); st.sidebar.info("Note: Transition settings primarily guide allocation...");
                end_year_label = st.session_state.actual_end_year if st.session_state.actual_end_year is not None else "End Year"; removal_help_text = (f"Target % of portfolio *volume* from Removals...in the final year ({end_year_label})...")
                default_removal_perc = int(st.session_state.get('removal_target_end_year', 0.8) * 100); removal_target_percent_slider = st.sidebar.slider(f"Target Removal Vol % for Year {end_year_label}", 0, 100, default_removal_perc, help=removal_help_text, key='removal_perc_slider_sidebar')
                st.session_state.removal_target_end_year = removal_target_percent_slider / 100.0; st.session_state.transition_speed = st.sidebar.slider("Transition Speed", 1, 10, st.session_state.get('transition_speed', 5), help="Speed of ramping up...", key='transition_speed_slider_sidebar')
                # Technical/Natural Split (unchanged)
                st.sidebar.markdown("### Removal Category Preference"); removal_preference_val = st.sidebar.slider("Technical vs Natural...", 1, 10, st.session_state.get('removal_preference_slider', 5), help="Influences split...", key='removal_pref_slider_sidebar')
                st.session_state['removal_preference_slider'] = removal_preference_val; tech_pref_norm = (removal_preference_val - 1) / 9.0; st.session_state.category_split = {'technical removal': tech_pref_norm, 'natural removal': 1.0 - tech_pref_norm}

        # Project Selection (unchanged)
        st.sidebar.markdown("## 3. Select Projects")
        if not project_names_list: st.sidebar.info("Upload data to see available projects.")
        else:
            default_selection = st.session_state.get('selected_projects', project_names_list); valid_default_selection = [p for p in default_selection if p in project_names_list];
            if not valid_default_selection: valid_default_selection = project_names_list
            st.session_state.selected_projects = st.sidebar.multiselect("Select projects for portfolio:", project_names_list, default=valid_default_selection, key='project_selector_sidebar')
            available_for_boost = [p for p in project_names_list if p in st.session_state.selected_projects]; priority_col_exists = data_df is not None and 'priority' in data_df.columns
            if priority_col_exists:
                current_favorites = st.session_state.get('favorite_projects_selection', []); valid_default_favorites = [fav for fav in current_favorites if fav in available_for_boost]
                if available_for_boost: st.session_state.favorite_projects_selection = st.sidebar.multiselect("Select Favorite Project (Boosts Priority):", available_for_boost, default=valid_default_favorites, max_selections=1, key='favorite_selector_sidebar')
                else: st.sidebar.info("Select projects above..."); st.session_state.favorite_projects_selection = []
            else: st.sidebar.info("No 'priority' column found..."); st.session_state.favorite_projects_selection = []

# ==================================
# Main Page Content
# ==================================
st.title("Carbon Portfolio Builder")
# --- Input Checks ---
if not st.session_state.get('data_loaded_successfully', False): st.info("👋 Welcome! Please upload project data via the sidebar to begin.")
elif not st.session_state.get('selected_projects'): st.warning("⚠️ Please select projects in the sidebar to build the portfolio.")
elif not st.session_state.get('selected_years'): st.warning("⚠️ Please select a valid Start and End Year in the sidebar.")
elif st.session_state.actual_start_year is None or st.session_state.actual_end_year is None : st.warning("⚠️ Invalid year range selected.")
else:
    # --- Check required state --- (unchanged)
    required_state_keys = ['working_data_full', 'selected_projects', 'selected_years', 'annual_targets', 'constraint_type', 'removal_target_end_year', 'transition_speed', 'category_split', 'favorite_projects_selection', 'actual_start_year', 'actual_end_year']
    required_state_keys_present = all(key in st.session_state for key in required_state_keys); required_state_values_present = all(st.session_state.get(key) is not None for key in required_state_keys if key not in ['favorite_projects_selection', 'annual_targets', 'selected_years']); valid_years_selected = bool(st.session_state.get('selected_years'))

    if required_state_keys_present and required_state_values_present and valid_years_selected:
        try:
            favorite_project_name = st.session_state.favorite_projects_selection[0] if st.session_state.favorite_projects_selection else None; current_constraint_type = st.session_state.constraint_type; summary_col_suffix = "Volume" if current_constraint_type == "Volume" else "Budget"; shortfall_col_name = f'{summary_col_suffix} Shortfall'; start_yr_alloc = st.session_state.actual_start_year; end_yr_alloc = st.session_state.actual_end_year

            # --- Run Allocation ---
            if start_yr_alloc is None or end_yr_alloc is None: st.error("Cannot run allocation: Planning horizon start/end years are not set.")
            else:
                with st.spinner("Calculating portfolio allocation..."):
                    # Pass necessary arguments to cached function
                    portfolio_results, summary_df = allocate_portfolio(
                        project_data=st.session_state.working_data_full,
                        selected_project_names=st.session_state.selected_projects,
                        selected_years=st.session_state.selected_years,
                        start_year_portfolio=start_yr_alloc,
                        end_year_portfolio=end_yr_alloc,
                        constraint_type=current_constraint_type,
                        # Pass hashable version of dicts if caching issues arise
                        annual_targets=st.session_state.annual_targets,
                        removal_target_percent_end_year=st.session_state.removal_target_end_year,
                        transition_speed=st.session_state.transition_speed,
                        category_split=st.session_state.category_split,
                        favorite_project=favorite_project_name,
                        priority_boost_percent=10 # Keep default or make configurable
                    )

                # --- Prepare Viz Data --- (unchanged)
                portfolio_data_list_viz = []
                if portfolio_results:
                    for year_viz, projects_list in portfolio_results.items():
                        if isinstance(projects_list, list):
                            for proj_info in projects_list:
                                if isinstance(proj_info, dict) and proj_info.get('allocated_volume', 0) > 0: portfolio_data_list_viz.append({'year': year_viz, 'project name': proj_info.get('project name', 'Unknown'), 'type': proj_info.get('type', 'Unknown'), 'volume': proj_info['allocated_volume'], 'price': proj_info.get('price_used', 0), 'cost': proj_info['allocated_cost']})
                portfolio_df_viz = pd.DataFrame(portfolio_data_list_viz) if portfolio_data_list_viz else pd.DataFrame()

                # --- Key Metrics Section (Revised Layout & Pie Chart) ---
                st.markdown("#### Portfolio Overview")
                container_css_class = "key-metrics-line"
                with st.container():
                    st.markdown(f'<div class="{container_css_class}">', unsafe_allow_html=True) # Div for CSS scoping
                    if not summary_df.empty and start_yr_alloc is not None and end_yr_alloc is not None:
                        total_volume = summary_df['Allocated Volume'].sum(); total_cost = summary_df['Allocated Cost'].sum(); overall_avg_price = (total_cost / total_volume) if total_volume > 0 else 0; total_shortfall = summary_df[shortfall_col_name].sum()
                        pie_data = pd.DataFrame()
                        if not portfolio_df_viz.empty and total_volume > 0:
                             pie_data = portfolio_df_viz.groupby('type')['volume'].sum().reset_index()
                             pie_data = pie_data[pie_data['volume'] > 0]
                             pie_data['percentage'] = (pie_data['volume'] / total_volume) * 100

                        # Use columns INSIDE the div for layout control
                        cols = st.columns([1.5, 1.5, 1.5, 1.5, 2], gap="medium", vertical_alignment="center")

                        with cols[0]:
                             # Use st.metric for consistent styling
                             st.metric(label="Horizon", value=f"{start_yr_alloc}–{end_yr_alloc}") # Use em dash
                        with cols[1]:
                            st.metric(label="Total Volume", value=f"{total_volume:,.0f} t")
                        with cols[2]:
                            st.metric(label="Total Cost", value=f"€{total_cost:,.2f}")
                        with cols[3]:
                            st.metric(label="Avg. Price", value=f"€{overall_avg_price:,.2f}/t")
                        with cols[4]:
                            # Display Pie Chart
                            if not pie_data.empty:
                                type_color_map_viz = {'technical removal': '#66BB6A', 'natural removal': '#AED581', 'reduction': '#388E3C'}
                                pie_data['type_display'] = pie_data['type'].apply(lambda x: x.replace('_', ' ').capitalize())
                                fig_pie = px.pie(pie_data,
                                                 values='volume', names='type_display', title="Volume Mix", # Added Title
                                                 color='type', color_discrete_map=type_color_map_viz, hole=0.3 )
                                fig_pie.update_traces(textposition='outside', textinfo='percent', textfont_size=11,
                                                      hovertemplate = "<b>%{label}</b><br>Volume: %{value:,.0f} t<br>%{percent:.1%}<extra></extra>",
                                                      insidetextorientation='radial')
                                fig_pie.update_layout(
                                    showlegend=True, # ** Show Legend **
                                    legend_title_text='Type',
                                    legend=dict(orientation="v", yanchor="top", y=0.9, xanchor="left", x=0.05, font=dict(size=10)),
                                    margin=dict(l=0, r=0, t=40, b=20), # ** Increased top/bottom margin **
                                    height=250, # ** Increased Height **
                                    title_font_size=14, title_y=0.97, title_x=0.5, title_xanchor='center', # Adjust title position slightly
                                    uniformtext_minsize=9, uniformtext_mode='hide' )
                                st.plotly_chart(fig_pie, use_container_width=True)
                            else: st.caption(" ") # Keep alignment

                    else:
                        st.markdown("<p style='text-align: center; font-style: italic;'>Run allocation to see overview.</p>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True) # Close the div


                # --- Plots and Tables Section ---
                st.header("📊 Portfolio Analysis & Visualization"); st.markdown("---")
                if portfolio_df_viz.empty: st.warning("No projects received allocation...")
                else:
                    type_color_map = {'technical removal': '#66BB6A', 'natural removal': '#AED581', 'reduction': '#388E3C'}; default_color = '#BDBDBD'
                    # Composition Plot (unchanged)
                    st.markdown("#### Portfolio Composition & Price Over Time");
                    if not portfolio_df_viz.empty: summary_plot_df = portfolio_df_viz.groupby(['year', 'type']).agg(volume=('volume', 'sum'), cost=('cost', 'sum')).reset_index()
                    else: summary_plot_df = pd.DataFrame()
                    fig_composition = make_subplots(specs=[[{"secondary_y": True}]]); y_axis_metric = 'volume' if current_constraint_type == 'Volume' else 'cost'; y_axis_label = 'Allocated Volume (tonnes)' if current_constraint_type == 'Volume' else 'Allocated Cost (€)'; y_axis_format = ',.0f' if current_constraint_type == 'Volume' else ',.2f'; y_axis_prefix = '' if current_constraint_type == 'Volume' else '€'; plot_type_order = ['reduction', 'natural removal', 'technical removal']; valid_types_in_results = summary_plot_df['type'].unique() if not summary_plot_df.empty else []
                    if not summary_plot_df.empty:
                        for type_name in plot_type_order:
                            if type_name in valid_types_in_results:
                                df_type = summary_plot_df[summary_plot_df['type'] == type_name]
                                if not df_type.empty and y_axis_metric in df_type.columns: fig_composition.add_trace(go.Bar(x=df_type['year'], y=df_type[y_axis_metric], name=type_name.replace('_', ' ').capitalize(), marker_color=type_color_map.get(type_name, default_color), hovertemplate=f'Year: %{{x}}<br>Type: {type_name.capitalize()}<br>{y_axis_label.split(" (")[0]}: {y_axis_prefix}%{{y:{y_axis_format}}}<extra></extra>'), secondary_y=False)
                    if not summary_df.empty:
                        if 'Avg. Price' in summary_df.columns: fig_composition.add_trace(go.Scatter(x=summary_df['Year'], y=summary_df['Avg. Price'], name='Avg Price (€/t)', mode='lines+markers', marker=dict(symbol='circle', size=8), line=dict(color='#1B5E20', width=3), hovertemplate='Year: %{x}<br>Avg Price: €%{y:,.2f}<extra></extra>'), secondary_y=True)
                        if 'Actual Removal Vol %' in summary_df.columns: fig_composition.add_trace(go.Scatter(x=summary_df['Year'], y=summary_df['Actual Removal Vol %'], name='Actual Removal Vol %', mode='lines+markers', line=dict(color='darkorange', dash='dash', width=2), marker=dict(symbol='star', size=8), hovertemplate='Year: %{x}<br>Actual Removal: %{y:.1f}%<extra></extra>'), secondary_y=True)
                        if 'Target Removal Vol %' in summary_df.columns: fig_composition.add_trace(go.Scatter(x=summary_df['Year'], y=summary_df['Target Removal Vol %'], name='Target Removal Vol %', mode='lines', line=dict(color='grey', dash='dot', width=1.5), hovertemplate='Year: %{x}<br>Target Removal: %{y:.1f}%<extra></extra>'), secondary_y=True)
                    y2_max_val = 105.0
                    if not summary_df.empty:
                        if 'Avg. Price' in summary_df.columns and not summary_df['Avg. Price'].isnull().all(): max_price_plot = summary_df['Avg. Price'].max(); y2_max_val = max(y2_max_val, max_price_plot * 1.15 if pd.notna(max_price_plot) else y2_max_val)
                        if 'Actual Removal Vol %' in summary_df.columns and not summary_df['Actual Removal Vol %'].isnull().all(): max_removal_plot = summary_df['Actual Removal Vol %'].max(); y2_max_val = max(y2_max_val, max_removal_plot * 1.15 if pd.notna(max_removal_plot) else y2_max_val)
                    fig_composition.update_layout(xaxis_title='Year', yaxis_title=y_axis_label, yaxis2_title='Avg Price (€/t) / Removal Vol %', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), barmode='stack', template="plotly_white", margin=dict(t=20, l=0, r=0, b=0), yaxis=dict(rangemode='tozero'), yaxis2=dict(rangemode='tozero', range=[0, y2_max_val]), hovermode="x unified")
                    if st.session_state.selected_years: fig_composition.update_xaxes(tickmode='array', tickvals=st.session_state.selected_years, dtick=1)
                    st.plotly_chart(fig_composition, use_container_width=True)

                    # --- ** FIX: Detailed Allocation View (No Treemap) ** ---
                    st.markdown("---")
                    st.markdown("#### 📄 Detailed Allocation View")

                    # Radio button and Detailed Table display
                    if portfolio_results and any(v for v in portfolio_results.values() if isinstance(v, list) and v):
                        years_with_allocations = sorted([yr for yr, details in portfolio_results.items() if isinstance(details, list) and details and any(d.get('allocated_volume',0) > 0 for d in details)])
                        df_all_years_agg = pd.DataFrame(); all_years_data_exists = False
                        if not portfolio_df_viz.empty: df_all_years_agg = portfolio_df_viz.groupby(['project name', 'type']).agg(allocated_volume=('volume', 'sum'), allocated_cost=('cost', 'sum')).reset_index(); df_all_years_agg['price_avg'] = df_all_years_agg.apply(lambda row: row['allocated_cost'] / row['allocated_volume'] if row['allocated_volume'] > 0 else 0, axis=1); df_all_years_agg = df_all_years_agg[['project name', 'type', 'allocated_volume', 'allocated_cost', 'price_avg']].sort_values(by=['type', 'project name']); all_years_data_exists = True
                        radio_options = [];
                        if all_years_data_exists: radio_options.append("All Years (Aggregated)")
                        if years_with_allocations: radio_options.extend([str(year) for year in years_with_allocations])

                        if radio_options:
                            # ** FIX: Updated Radio Label **
                            selected_view_detail = st.radio("Select View for Detailed Table:", radio_options, horizontal=True, key='detail_view_radio')

                            # --- REMOVED Treemap generation and display block ---

                            # --- Display Detailed Table ---
                            st.markdown("---") # Separator
                            if selected_view_detail == "All Years (Aggregated)":
                                st.markdown("##### Aggregated Allocation Table");
                                if not df_all_years_agg.empty: st.dataframe(df_all_years_agg.style.format({'allocated_volume': '{:,.0f}', 'allocated_cost': '€{:,.2f}', 'price_avg': '€{:,.2f}'}), hide_index=True, use_container_width=True)
                            else:
                                try:
                                    year_to_show = int(selected_view_detail); st.markdown(f"##### Detailed Allocation Table for {year_to_show}"); year_data_list = portfolio_results.get(year_to_show, [])
                                    if year_data_list:
                                        year_df_filtered = pd.DataFrame([p for p in year_data_list if p.get('allocated_volume',0) > 0])
                                        if not year_df_filtered.empty: year_df_sorted = year_df_filtered.sort_values(by=['type', 'project name']); display_cols_detail = ['project name', 'type', 'allocated_volume', 'allocated_cost', 'price_used', 'priority_applied', 'remaining_available']; existing_cols_detail = [col for col in display_cols_detail if col in year_df_sorted.columns]; st.dataframe(year_df_sorted[existing_cols_detail].style.format({'allocated_volume': '{:,.0f}', 'allocated_cost': '€{:,.2f}', 'price_used': '€{:,.2f}', 'priority_applied': '{:.4f}', 'remaining_available': '{:,.0f}'}), hide_index=True, use_container_width=True)
                                        else: st.info(f"No projects received allocation in {year_to_show}.")
                                    else: st.error(f"Internal error: Could not retrieve data for year {year_to_show}.")
                                except ValueError: st.error(f"Invalid year selected: {selected_view_detail}")
                        else: st.info("No detailed allocation data to display...")
                    else: st.info("Run allocation to view detailed results.")

                # Expander for full data download/view (unchanged)
                if not portfolio_df_viz.empty:
                    st.markdown("---");
                    with st.expander("Show Full Detailed Allocation Data (All Years & Projects)"):
                        portfolio_df_viz_sorted = portfolio_df_viz.sort_values(by=['project name', 'year']); csv_data = portfolio_df_viz_sorted[['year', 'project name', 'type', 'volume', 'price', 'cost']].to_csv(index=False).encode('utf-8'); st.download_button(label="Download Detailed Data as CSV", data=csv_data, file_name='detailed_portfolio_allocation.csv', mime='text/csv',)
                        display_portfolio_df_detail = portfolio_df_viz_sorted.copy(); cols_to_show_detail = ['year', 'project name', 'type', 'volume', 'price', 'cost']; existing_cols_detail = [col for col in cols_to_show_detail if col in display_portfolio_df_detail.columns]; st.dataframe(display_portfolio_df_detail[existing_cols_detail].style.format({'volume': '{:,.0f}', 'price': '€{:,.2f}', 'cost': '€{:,.2f}'}), hide_index=True, use_container_width=True)
        # Error Handling (unchanged)
        except ValueError as e: st.error(f"📉 Configuration or Allocation Error: {e}")
        except KeyError as e: st.error(f"📉 Data Error: Missing expected column or key: {e}. Check CSV format/names.")
        except UnboundLocalError as e: st.error(f"📉 Logic Error: Variable accessed before assignment: {e}")
        except Exception as e: st.error(f"📉 Unexpected error: {e}") # st.exception(e)

    # Footer (unchanged)
    try:
        local_tz_name = "Europe/Zurich"; local_tz = pytz.timezone(local_tz_name); current_date = datetime.datetime.now(local_tz); tz_name = current_date.strftime('%Z')
    except Exception as e: st.warning(f"Could not determine local timezone ({local_tz_name}), using UTC. Error: {e}"); current_date = datetime.datetime.now(datetime.timezone.utc); tz_name = "UTC" # Fallback
    st.markdown("---"); st.caption(f"Analysis generated on: {current_date.strftime('%Y-%m-%d %H:%M:%S')} ({tz_name})")
