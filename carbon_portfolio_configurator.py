# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import copy
import datetime # Import datetime for date check

# ==================================
# Configuration & Theming (Using Green Theme)
# ==================================
st.set_page_config(layout="wide")
# --- Enhanced Green Theme CSS (Slider Background Adjusted) ---
css = """
<style>
    /* Main App background */
    /* .stApp { background-color: #F1F8E9; } */

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #C8E6C9; padding-top: 2rem; } /* Lighter Green */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] .stText { color: #1B5E20; } /* Dark Green Text */
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .st-bq { color: #1B5E20 !important; }
    [data-testid="stSidebar"] .stButton>button { background-color: #4CAF50; color: white; border: none; }
    [data-testid="stSidebar"] .stButton>button:hover { background-color: #388E3C; color: white; }
    [data-testid="stSidebar"] .stNumberInput input, [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"], /* Adjusted selector for selectbox */
    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"],
    [data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] {
        border-color: #A5D6A7 !important; /* Light green border for widgets */
        background-color: #FFFFFF; /* Ensure white background for inputs */
        color: #1B5E20; /* Dark green text inside input */
    }
    /* Ensure dropdown list also has styling if needed - might require browser inspection */
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] { background: #C8E6C9; } /* ADJUSTED: Slider track matches sidebar bg */
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div:nth-child(3) { background-color: #388E3C; } /* Slider handle */
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] { background-color: #66BB6A; color: white; } /* Multiselect selected items */
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] span:first-child { background-color: #A5D6A7 !important; border-color: #388E3C !important; } /* Radio button circle */
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] span:first-child[aria-checked="true"] { background-color: #388E3C !important; } /* Radio button selected */


    /* Main content Margins */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; padding-left: 3rem; padding-right: 3rem; max-width: 100%; }

    /* Main Content Titles */
    h1, h2, h3, h4, h5, h6 { color: #1B5E20; } /* Dark Green */

    /* Dataframes */
    .stDataFrame { border: 1px solid #A5D6A7; }

</style>
"""
st.markdown(css, unsafe_allow_html=True)


# ==================================
# Allocation Function (Includes Volume/Budget Remainder Logic)
# ==================================
# (Function remains identical to the previous version - no changes needed here)
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
    """
    Allocates portfolio based on user selection and transition goals.
    Includes second pass for both Volume and Budget modes to better meet targets.
    """
    portfolio_details = {year: {} for year in selected_years}
    yearly_summary_list = []
    summary_col_suffix = "Volume" if constraint_type == "Volume" else "Budget"

    # --- 1. Filter Data and Validate Inputs ---
    if not selected_project_names:
         return {}, pd.DataFrame(columns=[
             'Year', f'Target {summary_col_suffix}', 'Allocated Volume', f'{summary_col_suffix} Shortfall',
             'Allocated Cost', 'Avg. Price', 'Actual Removal Vol %', 'Target Removal Vol %'])
    project_data_selected = project_data[project_data['project name'].isin(selected_project_names)].copy()
    if project_data_selected.empty:
         return {}, pd.DataFrame(columns=[
             'Year', f'Target {summary_col_suffix}', 'Allocated Volume', f'{summary_col_suffix} Shortfall',
             'Allocated Cost', 'Avg. Price', 'Actual Removal Vol %', 'Target Removal Vol %'])

    all_project_types_in_selection = project_data_selected['project type'].unique()
    if len(all_project_types_in_selection) == 0:
           st.warning("Selected projects have no valid project types.")
           return {}, pd.DataFrame(columns=[
             'Year', f'Target {summary_col_suffix}', 'Allocated Volume', f'{summary_col_suffix} Shortfall',
             'Allocated Cost', 'Avg. Price', 'Actual Removal Vol %','Target Removal Vol %'])

    is_reduction_selected = 'reduction' in all_project_types_in_selection
    total_years_duration = end_year_portfolio - start_year_portfolio
    min_allocation_chunk = 1

    # --- Input Validation and Data Type Conversion ---
    required_cols = ['project name', 'project type', 'priority']
    price_cols_needed = []
    volume_cols_needed = []
    for year in selected_years:
         price_col = f"price {year}"; volume_col = f"available volume {year}"
         if price_col in project_data_selected.columns and volume_col in project_data_selected.columns:
             required_cols.extend([price_col, volume_col])
             price_cols_needed.append(price_col); volume_cols_needed.append(volume_col)

    missing_cols = [col for col in ['project name', 'project type', 'priority'] if col not in project_data_selected.columns]
    if missing_cols:
         raise ValueError(f"Missing essential base columns in selected projects data: {', '.join(missing_cols)}")

    missing_year_data_cols = []
    for year in selected_years:
         price_col = f"price {year}"; volume_col = f"available volume {year}"
         if price_col not in project_data_selected.columns: missing_year_data_cols.append(price_col)
         if volume_col not in project_data_selected.columns: missing_year_data_cols.append(volume_col)

    if missing_year_data_cols:
          raise ValueError(f"Missing price/volume data for selected years (e.g., {', '.join(list(set(missing_year_data_cols))[:2])}...).")


    num_cols = ['priority'] + price_cols_needed + volume_cols_needed
    for col in num_cols:
         if col in project_data_selected.columns:
               project_data_selected[col] = pd.to_numeric(project_data_selected[col], errors='coerce')
               if col == 'priority': project_data_selected[col] = project_data_selected[col].fillna(0)
               if col.startswith("available volume"): project_data_selected[col] = project_data_selected[col].fillna(0).apply(lambda x: int(x) if pd.notna(x) else 0).clip(lower=0)
               if col.startswith("price"): project_data_selected[col] = project_data_selected[col].fillna(0.0).apply(lambda x: float(x) if pd.notna(x) else 0.0).clip(lower=0.0)


    # --- 2. Project Allocation (Loop through years) ---
    for year in selected_years:
        yearly_target = annual_targets.get(year, 0)
        price_col = f"price {year}"; volume_col = f"available volume {year}"
        allocation_dict_for_year = {}
        fractional_allocations = {}

        summary_template = {
             'Year': year, f'Target {summary_col_suffix}': yearly_target,
             'Allocated Volume': 0, f'{summary_col_suffix} Shortfall': 0,
             'Allocated Cost': 0, 'Avg. Price': 0,
             'Actual Removal Vol %': 0, 'Target Removal Vol %': 0 }

        if price_col not in project_data_selected.columns or volume_col not in project_data_selected.columns:
             st.warning(f"Data for year {year} ({price_col} or {volume_col}) not found in the uploaded file. Skipping allocation for this year.")
             yearly_summary_list.append(summary_template)
             portfolio_details[year] = []
             continue

        if yearly_target <= 0:
            yearly_summary_list.append(summary_template)
            portfolio_details[year] = []
            continue

        # --- Calculate Target % Mix for THIS year ---
        target_percentages = {}
        if is_reduction_selected:
             start_removal_pct = 0.10; end_removal_pct = removal_target_percent_end_year
             if total_years_duration <= 0: progress = 1.0
             else: progress = max(0, min(1, (year - start_year_portfolio) / total_years_duration))
             exponent = 0.1 + (11 - transition_speed) * 0.2; progress_factor = progress ** exponent
             target_removal_pct_year = start_removal_pct + (end_removal_pct - start_removal_pct) * progress_factor
             min_target_pct = min(start_removal_pct, end_removal_pct); max_target_pct = max(start_removal_pct, end_removal_pct)
             target_removal_pct_year = max(min_target_pct, min(max_target_pct, target_removal_pct_year))
             target_tech_pct_year = target_removal_pct_year * category_split.get('technical removal', 0)
             target_nat_pct_year = target_removal_pct_year * category_split.get('natural removal', 0)
             target_red_pct_year = max(0.0, 1.0 - target_tech_pct_year - target_nat_pct_year)
             target_percentages = {'reduction': target_red_pct_year, 'technical removal': target_tech_pct_year, 'natural removal': target_nat_pct_year}
        else: # No reduction selected
             tech_share = category_split.get('technical removal', 0); nat_share = category_split.get('natural removal', 0)
             total_removal_share = tech_share + nat_share
             tech_selected = 'technical removal' in all_project_types_in_selection; nat_selected = 'natural removal' in all_project_types_in_selection
             if total_removal_share > 1e-9:
                   tech_alloc = tech_share / total_removal_share if tech_selected else 0; nat_alloc = nat_share / total_removal_share if nat_selected else 0
                   total_alloc = tech_alloc + nat_alloc
                   if total_alloc > 1e-9:
                       target_percentages['technical removal'] = tech_alloc / total_alloc if tech_selected else 0
                       target_percentages['natural removal'] = nat_alloc / total_alloc if nat_selected else 0
                   else:
                       if tech_selected and nat_selected: target_percentages = {'technical removal': 0.5, 'natural removal': 0.5}
                       elif tech_selected: target_percentages['technical removal'] = 1.0
                       elif nat_selected: target_percentages['natural removal'] = 1.0
             else:
                  num_removal_types_selected = (1 if tech_selected else 0) + (1 if nat_selected else 0)
                  if num_removal_types_selected > 0:
                        equal_share = 1.0 / num_removal_types_selected
                        if tech_selected: target_percentages['technical removal'] = equal_share
                        if nat_selected: target_percentages['natural removal'] = equal_share
             target_percentages['reduction'] = 0.0

        current_sum = sum(target_percentages.values())
        if abs(current_sum - 1.0) > 1e-6 and current_sum > 0:
            norm_factor = 1.0 / current_sum
            target_percentages = {pt: share * norm_factor for pt, share in target_percentages.items()}

        target_removal_pct_for_summary = (target_percentages.get('technical removal', 0) + target_percentages.get('natural removal', 0)) * 100
        summary_template['Target Removal Vol %'] = target_removal_pct_for_summary

        # --- Initial Allocation within Types (First Pass) ---
        year_initial_total_allocated_vol = 0
        year_initial_total_allocated_cost = 0

        for project_type in all_project_types_in_selection:
            type_share = target_percentages.get(project_type, 0)
            if type_share <= 0: continue
            target_resource_for_type = yearly_target * type_share

            projects_in_type = project_data_selected[
                (project_data_selected['project type'] == project_type) &
                (project_data_selected[price_col] > 0) &
                (project_data_selected[volume_col] >= 0)
            ].copy()

            if projects_in_type.empty: continue

            # Priority Handling & Favorite Boost
            total_priority_in_type = projects_in_type['priority'].sum()
            if total_priority_in_type <= 0:
                num_projects = len(projects_in_type); equal_prio = 1.0 / num_projects if num_projects > 0 else 0
                projects_in_type['norm_prio_base'] = equal_prio
            else: projects_in_type['norm_prio_base'] = projects_in_type['priority'] / total_priority_in_type
            current_norm_priorities = projects_in_type.set_index('project name')['norm_prio_base'].to_dict()
            final_adjusted_priorities = current_norm_priorities.copy()
            if favorite_project and favorite_project in final_adjusted_priorities:
                 boost_factor = 1.0 + priority_boost_percent / 100.0
                 boosted_prios = {p: prio for p, prio in final_adjusted_priorities.items()}
                 boosted_prios[favorite_project] = boosted_prios[favorite_project] * boost_factor
                 total_boosted_prio = sum(boosted_prios.values())
                 if total_boosted_prio > 1e-9:
                     final_adjusted_priorities = {p: prio / total_boosted_prio for p, prio in boosted_prios.items()}

            # Budget Weight Calculation
            total_weight_for_type = 0; project_weights = {}
            if constraint_type == 'Budget':
                for _, project_row in projects_in_type.iterrows():
                    proj_name = project_row['project name']; p_prio_final = final_adjusted_priorities.get(proj_name, 0); p_price = project_row[price_col]
                    if p_price > 0:
                         current_weight = p_prio_final * p_price; project_weights[proj_name] = current_weight; total_weight_for_type += current_weight

            # Resource Allocation Loop (First Pass)
            for _, project_row in projects_in_type.iterrows():
                proj_name = project_row['project name']; proj_prio_final = final_adjusted_priorities.get(proj_name, 0)
                available_vol = project_row[volume_col]; price = project_row[price_col]
                allocated_volume_frac = 0.0
                allocated_volume_int = 0
                allocated_cost = 0

                if proj_name not in allocation_dict_for_year:
                     allocation_dict_for_year[proj_name] = {'project name': proj_name, 'type': project_type, 'allocated_volume': 0, 'allocated_cost': 0, 'price_used': price, 'priority_applied': proj_prio_final, 'initial_available': available_vol, 'remaining_available': available_vol }

                if price <= 0 or (proj_prio_final <= 0 and total_priority_in_type > 0) :
                     continue

                if constraint_type == 'Volume':
                    project_target_volume_frac = target_resource_for_type * proj_prio_final
                    allocated_volume_frac = min(project_target_volume_frac, available_vol)
                    fractional_allocations[proj_name] = allocated_volume_frac

                elif constraint_type == 'Budget':
                    project_target_volume_frac = 0
                    if total_weight_for_type > 1e-9:
                        normalized_weight = project_weights.get(proj_name, 0) / total_weight_for_type
                        project_target_budget = target_resource_for_type * normalized_weight
                        project_target_volume_frac = project_target_budget / price
                        allocated_volume_frac = min(project_target_volume_frac, available_vol)

                allocated_volume_int = int(max(0, math.floor(allocated_volume_frac)))

                if allocated_volume_int >= min_allocation_chunk and allocated_volume_int <= available_vol :
                    allocated_cost = allocated_volume_int * price
                    allocation_dict_for_year[proj_name].update({
                         'allocated_volume': allocated_volume_int,
                         'allocated_cost': allocated_cost,
                         'remaining_available': available_vol - allocated_volume_int
                         })
                    year_initial_total_allocated_vol += allocated_volume_int
                    year_initial_total_allocated_cost += allocated_cost


        # --- Remainder Distribution (Second Pass) ---
        year_final_total_allocated_vol = year_initial_total_allocated_vol
        year_final_total_allocated_cost = year_initial_total_allocated_cost

        if constraint_type == 'Volume':
            # Volume Remainder
            shortfall = int(round(yearly_target - year_initial_total_allocated_vol))
            if shortfall > 0:
                eligible_projects = []
                for proj_name, details in allocation_dict_for_year.items():
                    fractional_part = fractional_allocations.get(proj_name, 0.0) - details['allocated_volume']
                    if details['remaining_available'] > 0:
                        eligible_projects.append({'name': proj_name, 'fraction_lost': fractional_part, 'priority': details['priority_applied'], 'remaining_capacity': details['remaining_available'], 'price': details['price_used']})
                eligible_projects.sort(key=lambda x: (-x['fraction_lost'], -x['priority']))
                allocated_remainder = 0
                for project_info in eligible_projects:
                    if allocated_remainder >= shortfall: break
                    proj_name_rem = project_info['name']
                    can_allocate = min(project_info['remaining_capacity'], shortfall - allocated_remainder)
                    if can_allocate > 0:
                         price_rem = allocation_dict_for_year[proj_name_rem]['price_used']
                         allocation_dict_for_year[proj_name_rem]['allocated_volume'] += can_allocate
                         allocation_dict_for_year[proj_name_rem]['allocated_cost'] += can_allocate * price_rem
                         allocation_dict_for_year[proj_name_rem]['remaining_available'] -= can_allocate
                         year_final_total_allocated_vol += can_allocate
                         year_final_total_allocated_cost += can_allocate * price_rem
                         allocated_remainder += can_allocate

        elif constraint_type == 'Budget':
            # Budget Remainder
            budget_shortfall = yearly_target - year_initial_total_allocated_cost
            while budget_shortfall > 1e-6: # Tolerance
                cheapest_project_name = None
                cheapest_price = float('inf')
                candidate_found = False
                for proj_name, details in allocation_dict_for_year.items():
                    if details['remaining_available'] > 0 and details['price_used'] > 0:
                         if details['price_used'] < cheapest_price:
                             cheapest_price = details['price_used']
                             cheapest_project_name = proj_name
                             candidate_found = True
                if not candidate_found or cheapest_price > budget_shortfall or cheapest_price <= 0 :
                    break
                allocation_dict_for_year[cheapest_project_name]['allocated_volume'] += 1
                allocation_dict_for_year[cheapest_project_name]['allocated_cost'] += cheapest_price
                allocation_dict_for_year[cheapest_project_name]['remaining_available'] -= 1
                year_final_total_allocated_vol += 1
                year_final_total_allocated_cost += cheapest_price
                budget_shortfall -= cheapest_price


        # --- Final Calculations and Summary ---
        final_allocated_volume_this_year = sum(p['allocated_volume'] for p in allocation_dict_for_year.values())
        final_allocated_cost_this_year = sum(p['allocated_cost'] for p in allocation_dict_for_year.values())

        portfolio_details[year] = list(allocation_dict_for_year.values())

        summary_template['Allocated Volume'] = final_allocated_volume_this_year
        summary_template['Allocated Cost'] = final_allocated_cost_this_year
        summary_template['Avg. Price'] = (final_allocated_cost_this_year / final_allocated_volume_this_year) if final_allocated_volume_this_year > 0 else 0
        removal_vol = sum(p['allocated_volume'] for p in portfolio_details[year] if p['type'] != 'reduction')
        summary_template['Actual Removal Vol %'] = (removal_vol / final_allocated_volume_this_year * 100) if final_allocated_volume_this_year > 0 else 0

        if constraint_type == 'Volume':
             summary_template[f'{summary_col_suffix} Shortfall'] = max(0, int(round(yearly_target)) - final_allocated_volume_this_year)
        elif constraint_type == 'Budget':
             final_shortfall = yearly_target - final_allocated_cost_this_year
             summary_template[f'{summary_col_suffix} Shortfall'] = final_shortfall if final_shortfall > 1e-6 else 0.0

        yearly_summary_list.append(summary_template)

    yearly_summary_df = pd.DataFrame(yearly_summary_list)
    shortfall_col_name = f'{summary_col_suffix} Shortfall'
    if shortfall_col_name not in yearly_summary_df.columns:
         yearly_summary_df[shortfall_col_name] = 0.0 if constraint_type == 'Budget' else 0

    target_col_original = f'Target {summary_col_suffix}'
    if target_col_original in yearly_summary_df.columns:
         yearly_summary_df.rename(columns={target_col_original: 'Target Value'}, inplace=True)
    elif 'Target Value' not in yearly_summary_df.columns and 'Year' in yearly_summary_df.columns:
         yearly_summary_df['Target Value'] = yearly_summary_df['Year'].map(annual_targets).fillna(0)
    elif 'Target Value' not in yearly_summary_df.columns:
         yearly_summary_df['Target Value'] = 0


    display_cols = ['Year', 'Target Value', 'Allocated Volume', shortfall_col_name]
    display_cols.extend(['Allocated Cost', 'Avg. Price', 'Actual Removal Vol %', 'Target Removal Vol %'])
    final_cols = [col for col in display_cols if col in yearly_summary_df.columns]
    yearly_summary_df = yearly_summary_df[final_cols]

    return portfolio_details, yearly_summary_df


# ==================================
# Streamlit App Layout & Logic (REORDERED, VIZ CHANGES)
# ==================================
# Sidebar Section
with st.sidebar:
    st.markdown("## 1. Load Data")
    df_upload = st.file_uploader("Upload Project Data CSV", type="csv", key="uploader_sidebar")

    # Initialize session state
    # ... (Initialization remains the same) ...
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
    if 'master_target' not in st.session_state: st.session_state.master_target = None
    if 'data_loaded_successfully' not in st.session_state: st.session_state.data_loaded_successfully = False
    if 'years_slider_sidebar' not in st.session_state: st.session_state.years_slider_sidebar = 5
    if 'removal_preference_slider' not in st.session_state: st.session_state.removal_preference_slider = 5

    # --- Data Loading ---
    if df_upload:
        try:
            @st.cache_data
            def load_and_prepare_data(uploaded_file):
                 # Standardize column names on load
                 data = pd.read_csv(uploaded_file)
                 data.columns = data.columns.str.lower().str.strip().str.replace(' ', '_') # Standardize
                 core_cols_std = ['project_name', 'project_type', 'priority']
                 missing_essential = [col for col in core_cols_std if col not in data.columns]
                 if missing_essential: return None, f"Missing essential columns: {', '.join(missing_essential)}", [], [], []
                 numeric_prefixes_std = ['price_', 'available_volume_']
                 cols_to_convert = ['priority']
                 available_years = set()
                 price_cols_found = []
                 volume_cols_found = []
                 for col in data.columns:
                     for prefix in numeric_prefixes_std:
                         year_part = col[len(prefix):]
                         if col.startswith(prefix) and year_part.isdigit():
                             cols_to_convert.append(col); available_years.add(int(year_part))
                             if prefix == 'price_': price_cols_found.append(col)
                             else: volume_cols_found.append(col)
                             break
                 for col in list(set(cols_to_convert)):
                     if col in data.columns: data[col] = pd.to_numeric(data[col], errors='coerce')
                 data['priority'] = data['priority'].fillna(0)
                 for col in volume_cols_found: data[col] = data[col].fillna(0).apply(lambda x: max(0, int(x)) if pd.notna(x) else 0).clip(lower=0)
                 for col in price_cols_found: data[col] = data[col].fillna(0.0).apply(lambda x: max(0.0, float(x)) if pd.notna(x) else 0.0).clip(lower=0.0)
                 available_years = sorted(list(available_years))
                 if not available_years: return None, "No valid year data columns found (e.g., 'price_YYYY', 'available_volume_YYYY').", [], [], []
                 invalid_types_found = []
                 if 'project_type' in data.columns:
                     data['project_type'] = data['project_type'].str.lower().str.strip()
                     valid_types = ['reduction', 'technical removal', 'natural removal']
                     invalid_types_df = data[~data['project_type'].isin(valid_types)]
                     if not invalid_types_df.empty:
                           invalid_types_found = invalid_types_df['project_type'].unique().tolist()
                           data = data[data['project_type'].isin(valid_types)]
                 else: return None, "Missing 'project_type' column.", available_years, [], []
                 # RENAME COLUMNS TO DISPLAY FORMAT *LAST*
                 column_mapping_to_display = {col: col.replace('_', ' ') for col in data.columns}
                 data.rename(columns=column_mapping_to_display, inplace=True)
                 project_names_list = sorted(data['project name'].unique().tolist()) if 'project name' in data.columns else []
                 return data, None, available_years, project_names_list, invalid_types_found

            data, error_msg, available_years_in_data, project_names_list, invalid_types_found = load_and_prepare_data(df_upload)

            if invalid_types_found: st.sidebar.warning(f"Invalid project types ignored: {', '.join(invalid_types_found)}")
            if error_msg:
                st.sidebar.error(error_msg); st.session_state.data_loaded_successfully = False
                st.session_state.working_data_full = None; st.session_state.project_names = []; st.session_state.available_years_in_data = []
            else:
                st.session_state.project_names = project_names_list; st.session_state.available_years_in_data = available_years_in_data
                st.session_state.working_data_full = data; st.session_state.data_loaded_successfully = True; st.sidebar.success("Data loaded!")
                current_selection = st.session_state.get('selected_projects', [])
                valid_current_selection = [p for p in current_selection if p in project_names_list]
                if not valid_current_selection or not st.session_state.selected_projects:
                    st.session_state.selected_projects = project_names_list

        except Exception as e:
            st.sidebar.error(f"Error processing file: {e}"); st.session_state.data_loaded_successfully = False
            st.session_state.working_data_full = None

    # --- Sections 2 & 3 depend on successful load ---
    if st.session_state.get('data_loaded_successfully', False):
         data = st.session_state.working_data_full
         available_years_in_data = st.session_state.available_years_in_data
         project_names_list = st.session_state.project_names

         # --- Section 2: Portfolio Settings ---
         st.markdown("## 2. Portfolio Settings")
         if not available_years_in_data:
              st.sidebar.warning("No usable year data found in the uploaded file.")
         else:
              min_year = min(available_years_in_data); max_year_possible = max(available_years_in_data)
              years_max_slider = max(1, max_year_possible - min_year + 1)
              default_years = st.session_state.years_slider_sidebar
              if default_years > years_max_slider: default_years = years_max_slider

              years_to_plan = st.slider(f"Years to Plan (from {min_year})", 1, years_max_slider, default_years, key='years_slider_sidebar')
              start_year = min_year; end_year = start_year + years_to_plan - 1
              selected_years_range = list(range(start_year, end_year + 1))

              actual_years_present = []
              for year in selected_years_range:
                  price_col_std = f"price {year}"
                  vol_col_std = f"available volume {year}"
                  if price_col_std in data.columns and vol_col_std in data.columns:
                       actual_years_present.append(year)
              st.session_state.selected_years = actual_years_present

              if not st.session_state.selected_years:
                  st.sidebar.error(f"No price/volume data columns found in file for years {start_year}-{end_year}.")
              else:
                  st.session_state.actual_start_year = min(st.session_state.selected_years)
                  st.session_state.actual_end_year = max(st.session_state.selected_years)
                  st.sidebar.markdown(f"Planning Horizon: **{st.session_state.actual_start_year} - {st.session_state.actual_end_year}**")

                  # Constraint and Master Target (WITH CORRECTED SYNTAX)
                  st.session_state.constraint_type = st.radio("Constraint Type:", ('Volume', 'Budget'), index=['Volume', 'Budget'].index(st.session_state.get('constraint_type', 'Volume')), key='constraint_type_sidebar', horizontal=True)
                  constraint_type = st.session_state.constraint_type
                  master_target_value_from_state = st.session_state.get('master_target')

                  if constraint_type == 'Volume':
                       default_val_vol = 1000
                       if master_target_value_from_state is not None:
                           try: default_val_vol = int(float(master_target_value_from_state))
                           except (ValueError, TypeError): default_val_vol = 1000
                       master_target = st.number_input("Target Volume (tonnes per year):", min_value=0, step=100, value=default_val_vol, key='master_volume_sidebar')
                  else: # Budget
                       default_val_bud = 100000.0
                       if master_target_value_from_state is not None:
                           try: default_val_bud = float(master_target_value_from_state)
                           except (ValueError, TypeError): default_val_bud = 100000.0
                       master_target = st.number_input("Target Budget (€ per year):", min_value=0.0, step=1000.0, value=default_val_bud, format="%.2f", key='master_budget_sidebar')

                  st.session_state.master_target = master_target
                  st.session_state.annual_targets = {year: master_target for year in st.session_state.selected_years}

                  # Removal Transition Settings
                  st.sidebar.markdown("### Removal Volume Transition"); st.sidebar.info("Note: Transition settings apply only if 'Reduction' projects are selected.")
                  removal_help_text = (f"Target % of *volume* from Removals in the final year ({st.session_state.actual_end_year}). Starts at 10%. If 'Reduction' projects selected, this guides allocation. In 'Budget' mode, actual volume % may differ due to prices.")
                  removal_target_percent_slider = st.sidebar.slider(f"Target Removal Vol % for Year {st.session_state.actual_end_year}", 0, 100, int(st.session_state.get('removal_target_end_year', 0.8)*100), help=removal_help_text, key='removal_perc_slider_sidebar')
                  st.session_state.removal_target_end_year = removal_target_percent_slider / 100.0
                  st.session_state.transition_speed = st.sidebar.slider("Transition Speed", 1, 10, st.session_state.get('transition_speed', 5), help="Speed of ramping up removal % (1=Slow, 10=Fast), if Reductions selected.", key='transition_speed_slider_sidebar')

                  # Technical/Natural Split
                  st.sidebar.markdown("### Removal Category Preference")
                  removal_preference_val = st.sidebar.slider("Technical Removal Preference", 1, 10, st.session_state.get('removal_preference_slider', 5), help="Focus within removal target (1=More Natural, 10=More Technical)", key='removal_pref_slider_sidebar')
                  st.session_state['removal_preference_slider'] = removal_preference_val
                  tech_pref_norm = (removal_preference_val - 1) / 9.0
                  st.session_state.category_split = {'technical removal': tech_pref_norm, 'natural removal': 1.0 - tech_pref_norm}

                  # --- Section 3: Project Selection ---
                  st.sidebar.markdown("## 3. Select Projects")
                  st.session_state.selected_projects = st.sidebar.multiselect("Select projects for portfolio:", project_names_list, default=st.session_state.get('selected_projects', project_names_list), key='project_selector_sidebar')
                  available_for_boost = [p for p in project_names_list if p in st.session_state.selected_projects]
                  current_favorites = st.session_state.get('favorite_projects_selection', [])
                  valid_default_favorites = [fav for fav in current_favorites if fav in available_for_boost]
                  priority_col_exists = 'priority' in data.columns # Check display name 'priority'
                  if priority_col_exists:
                      if available_for_boost: st.session_state.favorite_projects_selection = st.sidebar.multiselect("Select Favorite Project (Boosts Priority):", available_for_boost, default=valid_default_favorites, max_selections=1, key='favorite_selector_sidebar')
                      else: st.sidebar.info("Select projects above to enable favorite boosting."); st.session_state.favorite_projects_selection = []
                  else: st.sidebar.info("No 'priority' column found; boosting disabled."); st.session_state.favorite_projects_selection = []


# ==================================
# Main Page Content (REORDERED, VIZ CHANGES)
# ==================================
st.title("Carbon Portfolio Builder")

# Check data/selection status
if not st.session_state.get('data_loaded_successfully', False):
    st.info("👋 Welcome! Please upload data via the sidebar to begin.")
elif not st.session_state.get('selected_projects'):
    st.warning("⚠️ Please select projects in the sidebar to calculate the portfolio.")
elif not st.session_state.get('selected_years'):
     st.warning("⚠️ No valid years selected based on slider and data availability. Adjust 'Years to Plan' in the sidebar.")
else:
    # Check required state keys
    required_state_keys = ['working_data_full', 'selected_projects', 'selected_years', 'annual_targets', 'constraint_type', 'removal_target_end_year', 'transition_speed', 'category_split', 'favorite_projects_selection', 'actual_start_year', 'actual_end_year']
    required_state_keys_present = all(key in st.session_state and st.session_state.get(key) is not None for key in required_state_keys if key != 'favorite_projects_selection')
    required_state_keys_present &= ('favorite_projects_selection' in st.session_state)
    valid_years_selected = bool(st.session_state.get('selected_years'))

    if required_state_keys_present and valid_years_selected:
        try:
            # --- Run Allocation ---
            favorite_project_name = st.session_state.favorite_projects_selection[0] if st.session_state.favorite_projects_selection else None
            current_constraint_type = st.session_state.constraint_type
            summary_col_suffix = "Volume" if current_constraint_type == "Volume" else "Budget"
            shortfall_col_name = f'{summary_col_suffix} Shortfall'

            with st.spinner("Calculating portfolio allocation..."):
                 portfolio_results, summary_df = allocate_portfolio(
                     project_data=st.session_state.working_data_full,
                     selected_project_names=st.session_state.selected_projects,
                     selected_years=st.session_state.selected_years,
                     start_year_portfolio=st.session_state.actual_start_year,
                     end_year_portfolio=st.session_state.actual_end_year,
                     constraint_type=current_constraint_type,
                     annual_targets=st.session_state.annual_targets,
                     removal_target_percent_end_year=st.session_state.removal_target_end_year,
                     transition_speed=st.session_state.transition_speed,
                     category_split=st.session_state.category_split,
                     favorite_project=favorite_project_name
                 )

            # --- Prepare Data for Visualization ---
            portfolio_data_list_viz = []
            if portfolio_results:
                 for year_viz, projects_list in portfolio_results.items():
                     if isinstance(projects_list, list):
                         for proj_info in projects_list:
                              if isinstance(proj_info, dict) and 'allocated_volume' in proj_info and 'allocated_cost' in proj_info:
                                    if proj_info.get('allocated_volume', 0) > 0 or proj_info.get('allocated_cost', 0) > 0:
                                        portfolio_data_list_viz.append({
                                             'year': year_viz, 'project name': proj_info.get('project name', 'Unknown'),
                                             'type': proj_info.get('type', 'Unknown'), 'volume': proj_info['allocated_volume'],
                                             'price': proj_info.get('price_used', 0), 'cost': proj_info['allocated_cost']})

            portfolio_df_viz = pd.DataFrame(portfolio_data_list_viz) if portfolio_data_list_viz else pd.DataFrame()


            # ============================================
            # Section 1: Portfolio Analysis & Visualization (MOVED UP)
            # ============================================
            st.header("📊 Portfolio Analysis & Visualization")
            st.markdown("---")

            if portfolio_df_viz.empty:
                 st.warning("No projects with positive allocation found. Cannot generate plots.")
            else:
                 type_color_map = {'technical removal': '#66BB6A', 'natural removal': '#AED581', 'reduction': '#388E3C'}
                 default_color = '#BDBDBD'

                 # --- Composition & Price Plot (DYNAMIC Y-AXIS) ---
                 st.markdown("#### Portfolio Composition & Price Over Time")
                 if not portfolio_df_viz.empty:
                     summary_plot_df = portfolio_df_viz.groupby(['year', 'type']).agg(volume=('volume', 'sum'), cost=('cost', 'sum')).reset_index()
                     price_summary = summary_plot_df.groupby('year').agg(total_volume=('volume', 'sum'), total_cost=('cost', 'sum')).reset_index()
                     price_summary['avg_price'] = price_summary.apply(lambda row: row['total_cost'] / row['total_volume'] if row['total_volume'] > 0 else 0, axis=1)
                 else:
                     summary_plot_df = pd.DataFrame()
                     price_summary = pd.DataFrame()

                 fig_composition = make_subplots(specs=[[{"secondary_y": True}]])
                 plot_type_order = ['reduction', 'natural removal', 'technical removal']
                 valid_types_in_results = portfolio_df_viz['type'].unique() if not portfolio_df_viz.empty else []
                 y_axis_metric = 'volume' if current_constraint_type == 'Volume' else 'cost'
                 y_axis_label = 'Volume (tonnes)' if current_constraint_type == 'Volume' else 'Allocated Cost (€)'
                 y_axis_format = '{:,.0f}' if current_constraint_type == 'Volume' else '€{:,.2f}'
                 y_axis_hover_label = 'Volume' if current_constraint_type == 'Volume' else 'Cost'

                 if not summary_plot_df.empty:
                     for type_name in plot_type_order:
                         if type_name in valid_types_in_results:
                             df_type = summary_plot_df[summary_plot_df['type'] == type_name]
                             if not df_type.empty and y_axis_metric in df_type.columns:
                                 fig_composition.add_trace(go.Bar(x=df_type['year'], y=df_type[y_axis_metric], name=type_name.capitalize(), marker_color=type_color_map.get(type_name, default_color), hovertemplate=f'Year: %{{x}}<br>Type: {type_name.capitalize()}<br>{y_axis_hover_label}: %{{y:{y_axis_format}}}<extra></extra>'), secondary_y=False)

                 if not price_summary.empty:
                      fig_composition.add_trace(go.Scatter(x=price_summary['year'], y=price_summary['avg_price'], name='Avg Price (€/t)', mode='lines+markers', marker=dict(symbol='circle', size=8), line=dict(color='#1B5E20', width=3), hovertemplate='Year: %{x}<br>Avg Price: €%{y:,.2f}<extra></extra>'), secondary_y=True)
                 if not summary_df.empty and 'Actual Removal Vol %' in summary_df.columns:
                     fig_composition.add_trace(go.Scatter(x=summary_df['Year'], y=summary_df['Actual Removal Vol %'], name='Actual Removal Vol %', mode='lines+markers', line=dict(color='darkorange', dash='dash'), marker=dict(symbol='star'), hovertemplate='Year: %{x}<br>Actual Removal: %{y:.1f}%<extra></extra>'), secondary_y=True)

                 y2_max = 105.0
                 if not price_summary.empty and 'avg_price' in price_summary.columns and not price_summary['avg_price'].isnull().all(): max_price = price_summary['avg_price'].max(); y2_max = max(y2_max, max_price * 1.1 if pd.notna(max_price) else y2_max)
                 if not summary_df.empty and 'Actual Removal Vol %' in summary_df.columns and not summary_df['Actual Removal Vol %'].isnull().all(): max_removal_pct = summary_df['Actual Removal Vol %'].max(); y2_max = max(y2_max, max_removal_pct * 1.1 if pd.notna(max_removal_pct) else y2_max)

                 fig_composition.update_layout(xaxis_title='Year', yaxis_title=y_axis_label, yaxis2_title='Avg Price (€/t) / Actual Removal %', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), barmode='stack', template="plotly_white", margin=dict(t=20, l=0, r=0, b=0), yaxis=dict(rangemode='tozero'), yaxis2=dict(rangemode='tozero', range=[0, y2_max]), hovermode="x unified")
                 if st.session_state.selected_years:
                     fig_composition.update_xaxes(tickmode='array', tickvals=st.session_state.selected_years, dtick=1)
                 st.plotly_chart(fig_composition, use_container_width=True)
                 st.markdown("---")

                 # --- Treemap Chart (Depends on Radio Button Selection Now) ---
                 st.markdown("#### Allocation Breakdown by Volume (Treemap)")
                 # The year selection for the treemap is now tied to the radio button selection below
                 # We will generate the treemap data *after* the radio button selection
                 # Placeholder for the treemap display area
                 treemap_placeholder = st.empty()


            # ============================================
            # Section 2: Portfolio Allocation Results (MOVED DOWN)
            # ============================================
            st.header("📈 Portfolio Allocation Results")
            st.markdown("#### Allocation Summary")

            if not summary_df.empty:
                target_col_label = f'Target {summary_col_suffix}'
                if 'Target Value' in summary_df.columns:
                     summary_df_display = summary_df.rename(columns={'Target Value': target_col_label})
                else:
                     summary_df_display = summary_df.copy()
                     if target_col_label not in summary_df_display.columns: summary_df_display[target_col_label] = 0

                format_dict = { 'Allocated Volume': '{:,.0f}', 'Allocated Cost': '€{:,.2f}', 'Avg. Price': '€{:,.2f}', 'Actual Removal Vol %': '{:.1f}%', 'Target Removal Vol %': '{:.1f}%' }
                if target_col_label in summary_df_display.columns: format_dict[target_col_label] = '{:,.0f}' if current_constraint_type == 'Volume' else '€{:,.2f}'
                if shortfall_col_name in summary_df_display.columns: format_dict[shortfall_col_name] = '{:,.0f}' if current_constraint_type == 'Volume' else '€{:,.2f}'

                st.dataframe(summary_df_display.style.format(format_dict), hide_index=True, use_container_width=True)

                # Shortfall Note
                if shortfall_col_name in summary_df_display.columns:
                    total_shortfall = summary_df_display[shortfall_col_name].sum()
                    is_shortfall_present = total_shortfall > (0 if current_constraint_type == 'Volume' else 1e-2)
                    if is_shortfall_present:
                         if current_constraint_type == 'Volume': st.caption(f"Note: '{shortfall_col_name}' indicates the target volume could not be fully met, likely due to insufficient total 'available volume'.")
                         else: st.caption(f"Note: '{shortfall_col_name}' shows remaining budget. This occurs if the next cheapest tonne costs more than the leftover budget or no volume remains.")

                # Budget Mode Transition Note
                if current_constraint_type == 'Budget': st.caption("Budget Mode Transition Note: The 'Target Removal Vol %' guides the *budget* split trend. The 'Actual Removal Vol %' achieved depends on relative project prices.")
            else: st.warning("Allocation resulted in an empty summary.")


            # --- Detailed Allocation View (Using Radio Buttons) ---
            st.markdown("#### 📄 Detailed Allocation View")
            if portfolio_results and any(v for v in portfolio_results.values() if isinstance(v, list) and v):
                years_with_allocations = sorted([yr for yr in st.session_state.selected_years if yr in portfolio_results and isinstance(portfolio_results[yr], list) and portfolio_results[yr]])

                # Prepare "All Years" Data for table view
                if not portfolio_df_viz.empty:
                     df_all_years_agg = portfolio_df_viz.groupby(['project name', 'type']).agg(
                         allocated_volume=('volume', 'sum'),
                         allocated_cost=('cost', 'sum')
                     ).reset_index()
                     df_all_years_agg['price_avg'] = df_all_years_agg.apply(
                         lambda row: row['allocated_cost'] / row['allocated_volume'] if row['allocated_volume'] > 0 else 0, axis=1
                     )
                     df_all_years_agg = df_all_years_agg[['project name', 'type', 'allocated_volume', 'allocated_cost', 'price_avg']].sort_values(by='project name')
                     all_years_data_exists = True
                else:
                     df_all_years_agg = pd.DataFrame()
                     all_years_data_exists = False

                # --- Create Radio Buttons for Year Selection ---
                radio_options = []
                if all_years_data_exists: radio_options.append("All Years")
                if years_with_allocations: radio_options.extend([str(year) for year in years_with_allocations])

                if radio_options: # Only show radio buttons if there's data
                     selected_year_detail = st.radio(
                         "Select Year for Detailed View & Treemap:",
                         radio_options,
                         horizontal=True,
                         key='detail_year_radio'
                     )

                     # --- Display Detailed Table based on Radio Selection ---
                     if selected_year_detail == "All Years":
                         st.markdown("##### Aggregated Allocation (All Selected Years)")
                         if not df_all_years_agg.empty:
                             st.dataframe(df_all_years_agg.style.format({
                                 'allocated_volume': '{:,.0f}', 'allocated_cost': '€{:,.2f}', 'price_avg': '€{:,.2f}'
                             }), hide_index=True, use_container_width=True)
                         # else: Already handled by radio options check
                     else: # Specific Year Selected
                         year_to_show = int(selected_year_detail)
                         st.markdown(f"##### Detailed Allocation for {year_to_show}")
                         year_data_list = portfolio_results.get(year_to_show, [])
                         if year_data_list:
                             year_df = pd.DataFrame(year_data_list).sort_values(by='project name')
                             display_cols_detail = ['project name', 'type', 'allocated_volume', 'allocated_cost', 'price_used', 'priority_applied']
                             existing_cols_detail = [col for col in display_cols_detail if col in year_df.columns]
                             display_df = year_df[existing_cols_detail]
                             st.dataframe(display_df.style.format({'allocated_volume': '{:,.0f}', 'allocated_cost': '€{:,.2f}', 'price_used': '€{:,.2f}', 'priority_applied': '{:.4f}'}), hide_index=True, use_container_width=True)
                         # else: Should not happen if year is in radio options

                     # --- Generate and Display Treemap based on Radio Selection ---
                     # This code now runs *after* the radio button selection
                     if not portfolio_df_viz.empty:
                          if selected_year_detail == "All Years":
                               df_treemap = portfolio_df_viz.groupby(['type', 'project name']).agg(volume=('volume', 'sum')).reset_index()
                          else:
                               year_to_show_treemap = int(selected_year_detail)
                               df_treemap = portfolio_df_viz[portfolio_df_viz['year'] == year_to_show_treemap].groupby(['type', 'project name']).agg(volume=('volume', 'sum')).reset_index()

                          df_treemap = df_treemap[df_treemap['volume'] > 0]

                          if not df_treemap.empty:
                               fig_treemap = px.treemap(df_treemap, path=[px.Constant("All Projects"), 'type', 'project name'],
                                                        values='volume', color='type',
                                                        color_discrete_map=type_color_map, title=None,
                                                        hover_data={'type': False, 'project name': True, 'volume': ':.0f'})
                               fig_treemap.update_traces(
                                   hovertemplate='<b>%{label}</b><br>Type: %{parent}<br>Volume: %{value:,.0f}<br>% of Total: %{percentRoot:.1%}<extra></extra>',
                                   textinfo='label+value',
                                   textfont_size=16 # Increased text size
                                   )
                               fig_treemap.update_layout(
                                    margin=dict(t=20, l=0, r=0, b=0),
                                    height=600  # Experiment with different heights (e.g., 600, 700, 800)
                                )
                               treemap_placeholder.plotly_chart(fig_treemap, use_container_width=True)
                          else:
                               treemap_placeholder.info(f"No allocation data to display in Treemap for {selected_year_detail}.")
                     else:
                          treemap_placeholder.info("No allocation data available to generate Treemap.") # Should not happen if radio buttons shown


                else: # No radio options means no data
                    st.info("No detailed allocation data to display.")


            else: st.info("No projects allocated in any selected year.")

            # --- Moved Detailed Allocation Data Table Expander ---
            # This section is now at the end


        except ValueError as e: st.error(f"Configuration or Allocation Error: {e}")
        except KeyError as e: st.error(f"Data Error: Missing expected column or key: {e}. Check CSV format/selections.")
        except Exception as e: st.error(f"An unexpected error occurred: {e}") # st.exception(e) # For debugging


    # ============================================
    # Section 3: Detailed Data Table (MOVED TO BOTTOM)
    # ============================================
    # Check if data exists before showing expander
    if not portfolio_df_viz.empty:
         st.markdown("---") # Add separator
         with st.expander("Show Detailed Allocation Data Table (Sorted by Project, then Year)"):
             # Sort data by Project Name, then Year
             portfolio_df_viz_sorted = portfolio_df_viz.sort_values(by=['project name', 'year'])

             # Prepare data for download (use the sorted, unformatted data)
             csv_data = portfolio_df_viz_sorted[[
                 'year', 'project name', 'type', 'volume', 'price', 'cost' # Select final columns
                 ]].to_csv(index=False).encode('utf-8')

             st.download_button(
                label="Download Detailed Data as CSV",
                data=csv_data,
                file_name='detailed_portfolio_allocation.csv',
                mime='text/csv',
             )

             # Format data for display AFTER preparing download data
             display_portfolio_df = portfolio_df_viz_sorted.copy()
             if 'volume' in display_portfolio_df.columns: display_portfolio_df['volume'] = display_portfolio_df['volume'].map('{:,.0f}'.format)
             if 'price' in display_portfolio_df.columns: display_portfolio_df['price'] = display_portfolio_df['price'].map('€{:,.2f}'.format)
             if 'cost' in display_portfolio_df.columns: display_portfolio_df['cost'] = display_portfolio_df['cost'].map('€{:,.2f}'.format)

             # Select columns for display
             cols_to_show = ['year', 'project name', 'type', 'volume', 'price', 'cost']
             existing_cols = [col for col in cols_to_show if col in display_portfolio_df.columns]
             st.dataframe(display_portfolio_df[existing_cols], hide_index=True, use_container_width=True)

    # Simple footer
    current_date = datetime.datetime.now()
    st.caption(f"Analysis generated on: {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
