# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
# import copy # No longer needed
import datetime # Import datetime for date check
import pytz # For timezone handling
import traceback # For detailed error logging

# ==================================
# Configuration & Theming (Using Green Theme)
# ==================================
st.set_page_config(layout="wide")
# --- Enhanced Green Theme CSS ---
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
    [data-testid="stSidebar"] .stNumberInput input, [data-testid="stSidebar"] .stSelectbox select,
    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"],
    [data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] {
        border-color: #A5D6A7 !important; /* Light green border for widgets */
    }
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] { background: #A5D6A7; }
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div:nth-child(3) { background-color: #388E3C; }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] { background-color: #66BB6A; color: white; }
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] span:first-child { background-color: #A5D6A7 !important; border-color: #388E3C !important; }
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] span:first-child[aria-checked="true"] { background-color: #388E3C !important; }


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
# Allocation Function (User Defined Logic - REVISED with Adjustment Step)
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
    min_target_fulfillment_percent: float = 0.95 # Target fulfillment goal (e.g., 95%)
) -> tuple[dict, pd.DataFrame]:
    """
    Allocates portfolio based on user selection, constraints, and transition goals.
    (Function documentation remains the same)
    """
    portfolio_details = {year: [] for year in selected_years}
    yearly_summary_list = []
    min_allocation_chunk = 1 # Minimum volume to allocate in one go (integer tonnes)

    # --- 1. Filter Data to Selected Projects ONLY ---
    if not selected_project_names:
        st.warning("No projects selected for allocation.")
        return {}, pd.DataFrame(columns=['Year', f'Target {constraint_type}', 'Allocated Volume', 'Allocated Cost', 'Avg. Price', 'Actual Removal Vol %', 'Target Removal Vol %'])

    project_data_selected = project_data[project_data['project name'].isin(selected_project_names)].copy()
    if project_data_selected.empty:
        st.warning("Selected projects not found in the loaded data.")
        return {}, pd.DataFrame(columns=['Year', f'Target {constraint_type}', 'Allocated Volume', 'Allocated Cost', 'Avg. Price', 'Actual Removal Vol %', 'Target Removal Vol %'])

    all_project_types_in_selection = project_data_selected['project type'].unique()
    if len(all_project_types_in_selection) == 0:
         st.warning("Selected projects have no valid project types ('reduction', 'technical removal', 'natural removal').")
         return {}, pd.DataFrame(columns=['Year', f'Target {constraint_type}', 'Allocated Volume', 'Allocated Cost', 'Avg. Price', 'Actual Removal Vol %', 'Target Removal Vol %'])

    is_reduction_selected = 'reduction' in all_project_types_in_selection
    total_years_duration = end_year_portfolio - start_year_portfolio

    # --- Input Validation and Data Type Conversion ---
    required_base_cols = ['project name', 'project type', 'priority']
    price_cols_needed = []
    volume_cols_needed = []
    for year in selected_years:
        price_col = f"price {year}"; volume_col = f"available volume {year}"
        price_cols_needed.append(price_col); volume_cols_needed.append(volume_col)

    missing_base_cols = [col for col in required_base_cols if col not in project_data_selected.columns]
    if missing_base_cols: raise ValueError(f"Missing required base columns: {', '.join(missing_base_cols)}")

    missing_year_data_cols = []
    for year in selected_years:
        price_col = f"price {year}"; volume_col = f"available volume {year}"
        if price_col not in project_data_selected.columns: missing_year_data_cols.append(price_col)
        if volume_col not in project_data_selected.columns: missing_year_data_cols.append(volume_col)
    if missing_year_data_cols:
         years_affected = sorted(list(set(int(c.split()[-1]) for c in missing_year_data_cols if c.split()[-1].isdigit())))
         raise ValueError(f"Missing price/volume data for year(s): {', '.join(map(str, years_affected))}. Check data for selected horizon.")

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
        year_total_allocated_vol = 0; year_total_allocated_cost = 0
        summary_template = {'Year': year, f'Target {constraint_type}': yearly_target, 'Allocated Volume': 0, 'Allocated Cost': 0, 'Avg. Price': 0, 'Actual Removal Vol %': 0, 'Target Removal Vol %': 0 }
        if yearly_target <= 0:
            yearly_summary_list.append(summary_template); portfolio_details[year] = []; continue

        # --- Calculate Target % Mix ---
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
        else:
            tech_share_pref = category_split.get('technical removal', 0); nat_share_pref = category_split.get('natural removal', 0)
            total_removal_pref_share = tech_share_pref + nat_share_pref
            tech_selected = 'technical removal' in all_project_types_in_selection; nat_selected = 'natural removal' in all_project_types_in_selection
            tech_alloc = 0.0; nat_alloc = 0.0
            if total_removal_pref_share > 1e-9:
                if tech_selected: tech_alloc = tech_share_pref / total_removal_pref_share
                if nat_selected: nat_alloc = nat_share_pref / total_removal_pref_share
            else:
                 num_removal = (1 if tech_selected else 0) + (1 if nat_selected else 0)
                 if num_removal > 0: equal_share = 1.0 / num_removal
                 else: equal_share = 0
                 if tech_selected: tech_alloc = equal_share
                 if nat_selected: nat_alloc = equal_share
            total_alloc = tech_alloc + nat_alloc
            if total_alloc > 1e-9:
                target_percentages['technical removal'] = tech_alloc / total_alloc if tech_selected else 0
                target_percentages['natural removal'] = nat_alloc / total_alloc if nat_selected else 0
            else: target_percentages['technical removal'] = 0.0; target_percentages['natural removal'] = 0.0
            target_percentages['reduction'] = 0.0

        current_sum = sum(target_percentages.values())
        if abs(current_sum - 1.0) > 1e-6 and current_sum > 0:
            norm_factor = 1.0 / current_sum; target_percentages = {pt: share * norm_factor for pt, share in target_percentages.items()}
        summary_template['Target Removal Vol %'] = (target_percentages.get('technical removal', 0) + target_percentages.get('natural removal', 0)) * 100

        # --- 3. Initial Allocation ---
        projects_for_year_df = project_data_selected[(project_data_selected[price_col] > 0) & (project_data_selected[volume_col] >= min_allocation_chunk)].copy()
        projects_for_year_df['initial_allocated_volume'] = 0; projects_for_year_df['initial_allocated_cost'] = 0.0; projects_for_year_df['final_priority'] = np.nan
        if projects_for_year_df.empty:
            yearly_summary_list.append(summary_template); portfolio_details[year] = []; continue

        for project_type in all_project_types_in_selection:
            type_share = target_percentages.get(project_type, 0)
            if type_share <= 0: continue
            target_resource_for_type = yearly_target * type_share
            projects_in_type = projects_for_year_df[projects_for_year_df['project type'] == project_type].copy()
            if projects_in_type.empty: continue

            total_priority = projects_in_type['priority'].sum()
            if total_priority <= 0: num_proj = len(projects_in_type); projects_in_type['norm_prio_base'] = (1.0 / num_proj) if num_proj > 0 else 0
            else: projects_in_type['norm_prio_base'] = projects_in_type['priority'] / total_priority
            current_norm_priorities = projects_in_type.set_index('project name')['norm_prio_base'].to_dict()
            final_adjusted_priorities = current_norm_priorities.copy()

            if favorite_project and favorite_project in final_adjusted_priorities:
                fav_prio = current_norm_priorities[favorite_project]; boost = priority_boost_percent / 100.0
                increase = fav_prio * boost; new_fav = fav_prio + increase
                others = [p for p in current_norm_priorities if p != favorite_project]; sum_others = sum(current_norm_priorities[p] for p in others)
                temp_prios = {favorite_project: new_fav}
                if sum_others > 1e-9:
                    reduc_factor = increase / sum_others
                    for p_name in others: temp_prios[p_name] = max(0, current_norm_priorities[p_name] * (1 - reduc_factor))
                else:
                    for p_name in others: temp_prios[p_name] = 0
                total_final = sum(temp_prios.values())
                if total_final > 1e-9: final_adjusted_priorities = {p: prio / total_final for p, prio in temp_prios.items()}
                elif favorite_project in temp_prios: final_adjusted_priorities = {favorite_project: 1.0}
                else: final_adjusted_priorities = current_norm_priorities

            project_weights = {}; total_weight = 0
            if constraint_type == 'Budget':
                for _, row in projects_in_type.iterrows():
                    name = row['project name']; prio = final_adjusted_priorities.get(name, 0); price = row[price_col]
                    weight = prio * price if price > 0 else 0; project_weights[name] = weight; total_weight += weight

            for index, row in projects_in_type.iterrows():
                name = row['project name']; prio = final_adjusted_priorities.get(name, 0); vol = row[volume_col]; price = row[price_col]
                alloc_vol = 0; alloc_cost = 0
                projects_for_year_df.loc[projects_for_year_df['project name'] == name, 'final_priority'] = prio
                if prio <= 0 or price <= 0 or vol < min_allocation_chunk: continue
                if constraint_type == 'Volume': target_vol = target_resource_for_type * prio; alloc_vol = min(target_vol, vol)
                elif constraint_type == 'Budget':
                    if total_weight > 1e-9: weight_norm = project_weights.get(name, 0) / total_weight; target_budg = target_resource_for_type * weight_norm; target_vol = target_budg / price; alloc_vol = min(target_vol, vol)
                    else: alloc_vol = 0
                alloc_vol = int(max(0, math.floor(alloc_vol)))
                if alloc_vol >= min_allocation_chunk:
                    alloc_cost = alloc_vol * price
                    projects_for_year_df.loc[projects_for_year_df['project name'] == name, 'initial_allocated_volume'] = alloc_vol
                    projects_for_year_df.loc[projects_for_year_df['project name'] == name, 'initial_allocated_cost'] = alloc_cost
                    year_total_allocated_vol += alloc_vol; year_total_allocated_cost += alloc_cost

        # --- 4. Adjustment Step ---
        target_thresh = yearly_target * min_target_fulfillment_percent; current_metric = year_total_allocated_cost if constraint_type == 'Budget' else year_total_allocated_vol
        if current_metric < target_thresh and yearly_target > 0:
            needed = target_thresh - current_metric
            projects_for_year_df['remaining_volume'] = projects_for_year_df[volume_col] - projects_for_year_df['initial_allocated_volume']
            candidates = projects_for_year_df[(projects_for_year_df['remaining_volume'] >= min_allocation_chunk) & (projects_for_year_df[price_col] > 0)].sort_values(by='priority', ascending=False).copy()
            for index, row in candidates.iterrows():
                if needed <= 1e-6: break
                name = row['project name']; price = row[price_col]; avail_add = row['remaining_volume']
                vol_added = 0; cost_added = 0
                if constraint_type == 'Volume':
                    add = int(math.floor(min(avail_add, needed)))
                    if add >= min_allocation_chunk: vol_added = add; cost_added = add * price; needed -= add
                    else: continue
                elif constraint_type == 'Budget':
                    max_afford = int(math.floor(needed / price)); add = min(avail_add, max_afford)
                    if add >= min_allocation_chunk:
                        cost_try = add * price
                        if cost_try <= needed * 1.1 or cost_try < price * min_allocation_chunk * 1.5: vol_added = add; cost_added = cost_try; needed -= cost_added
                        else: continue
                    else: continue
                if vol_added > 0:
                    projects_for_year_df.loc[index, 'initial_allocated_volume'] += vol_added; projects_for_year_df.loc[index, 'initial_allocated_cost'] += cost_added
                    projects_for_year_df.loc[index, 'remaining_volume'] -= vol_added; year_total_allocated_vol += vol_added; year_total_allocated_cost += cost_added

        # --- 5. Finalize ---
        final_list = []; final_df = projects_for_year_df[projects_for_year_df['initial_allocated_volume'] >= min_allocation_chunk].copy()
        for index, row in final_df.iterrows():
            # Ensure price_col is valid for the current year before accessing row[price_col]
            current_price = row.get(price_col, None) # Use .get with default None
            final_list.append({
                'project name': row['project name'],
                'type': row['project type'],
                'allocated_volume': row['initial_allocated_volume'],
                'allocated_cost': row['initial_allocated_cost'],
                'price_used': current_price, # Use the retrieved price
                'priority_applied': row['final_priority']
             })
        portfolio_details[year] = final_list
        summary_template['Allocated Volume'] = year_total_allocated_vol; summary_template['Allocated Cost'] = year_total_allocated_cost
        summary_template['Avg. Price'] = (year_total_allocated_cost / year_total_allocated_vol) if year_total_allocated_vol > 0 else 0
        removal_vol = sum(p['allocated_volume'] for p in final_list if p['type'] != 'reduction')
        summary_template['Actual Removal Vol %'] = (removal_vol / year_total_allocated_vol * 100) if year_total_allocated_vol > 0 else 0
        yearly_summary_list.append(summary_template)

    yearly_summary_df = pd.DataFrame(yearly_summary_list)
    if constraint_type == 'Budget':
        check = yearly_summary_df.copy(); check['Target Budget'] = check['Year'].map(annual_targets)
        over = check[check['Allocated Cost'] > check['Target Budget'] * 1.001]
        if not over.empty: st.warning(f"Budget target may be slightly exceeded in years {over['Year'].tolist()} due to adjustments/chunk sizes.")
    return portfolio_details, yearly_summary_df


# ==================================
# Streamlit App Layout & Logic
# ==================================

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 1. Load Data")
    df_upload = st.file_uploader("Upload Project Data CSV", type="csv", key="uploader_sidebar")

    default_values = {
        'working_data_full': None, 'selected_years': [], 'selected_projects': [], 'project_names': [],
        'favorite_projects_selection': [], 'actual_start_year': None, 'actual_end_year': None,
        'available_years_in_data': [], 'constraint_type': 'Volume', 'removal_target_end_year': 0.8,
        'transition_speed': 5, 'category_split': {'technical removal': 0.5, 'natural removal': 0.5},
        'annual_targets': {}, 'master_target': None, 'data_loaded_successfully': False,
        'years_slider_sidebar': 5, 'min_fulfillment_perc': 95, 'removal_preference_slider': 5
    }
    for key, default_value in default_values.items():
        if key not in st.session_state: st.session_state[key] = default_value

    if df_upload:
        try:
            @st.cache_data
            def load_and_prepare_data(uploaded_file):
                # ... (Keep the robust loading function) ...
                try:
                    data = pd.read_csv(uploaded_file)
                    data.columns = data.columns.str.lower().str.strip().str.replace(' ', '_')
                except Exception as read_error: return None, f"Error reading CSV: {read_error}", [], [], []
                core_cols_std = ['project_name', 'project_type', 'priority']
                missing_essential = [col for col in core_cols_std if col not in data.columns]
                if missing_essential: return None, f"Missing essential columns: {', '.join(missing_essential)}", [], [], []
                numeric_prefixes_std = ['price_', 'available_volume_']
                cols_to_convert = ['priority']; available_years = set(); year_data_cols = []
                for col in data.columns:
                    for prefix in numeric_prefixes_std:
                        year_part = col[len(prefix):]
                        if col.startswith(prefix) and year_part.isdigit():
                            cols_to_convert.append(col); year_data_cols.append(col); available_years.add(int(year_part)); break
                if not available_years:
                     has_price = any(c.startswith('price_') for c in data.columns); has_vol = any(c.startswith('available_volume_') for c in data.columns)
                     err = "No 'price_YYYY'/'available_volume_YYYY' cols found." if not has_price and not has_vol else "Found price/vol cols, but couldn't extract years. Check naming."
                     return None, err, [], [], []
                for col in list(set(cols_to_convert)):
                    if col in data.columns: data[col] = pd.to_numeric(data[col], errors='coerce')
                data['priority'] = data['priority'].fillna(0)
                for col in data.columns:
                     if col.startswith('available_volume_') and col in year_data_cols: data[col] = data[col].fillna(0).apply(lambda x: max(0, int(x)) if pd.notna(x) else 0).clip(lower=0)
                     elif col.startswith('price_') and col in year_data_cols: data[col] = data[col].fillna(0.0).apply(lambda x: max(0.0, float(x)) if pd.notna(x) else 0.0).clip(lower=0)
                available_years = sorted(list(available_years))
                invalid_types_found = []
                if 'project_type' in data.columns:
                    data['project_type'] = data['project_type'].astype(str).str.lower().str.strip()
                    valid_types = ['reduction', 'technical removal', 'natural removal']
                    invalid_types_df = data[~data['project_type'].isin(valid_types)]
                    if not invalid_types_df.empty:
                         invalid_types_found = invalid_types_df['project_type'].unique().tolist()
                         data = data[data['project_type'].isin(valid_types)].copy()
                else: return None, "Missing 'project_type' column.", available_years, [], []
                column_mapping_to_display = {col: col.replace('_', ' ') for col in data.columns}
                data.rename(columns=column_mapping_to_display, inplace=True)
                project_names_list = sorted(data['project name'].unique().tolist()) if 'project name' in data.columns else []
                return data, None, available_years, project_names_list, invalid_types_found

            data, error_msg, available_years_in_data, project_names_list, invalid_types_found = load_and_prepare_data(df_upload)
            if invalid_types_found: st.sidebar.warning(f"Ignored rows with invalid types: {', '.join(invalid_types_found)}. Valid: 'reduction', 'technical removal', 'natural removal'.")
            if error_msg:
                st.sidebar.error(error_msg); st.session_state.data_loaded_successfully = False
                st.session_state.working_data_full = None; st.session_state.project_names = []; st.session_state.available_years_in_data = []; st.session_state.selected_projects = []
            else:
                st.session_state.project_names = project_names_list; st.session_state.available_years_in_data = available_years_in_data
                st.session_state.working_data_full = data; st.session_state.data_loaded_successfully = True; st.sidebar.success("Data loaded!")
                current_selection = st.session_state.get('selected_projects', [])
                valid_current_selection = [p for p in current_selection if p in project_names_list]
                if not valid_current_selection: st.session_state.selected_projects = project_names_list
                else: st.session_state.selected_projects = valid_current_selection
        except Exception as e:
            st.sidebar.error(f"Error processing file: {e}"); st.session_state.data_loaded_successfully = False
            st.session_state.working_data_full = None; st.session_state.project_names = []; st.session_state.available_years_in_data = []; st.session_state.selected_projects = []

    if st.session_state.get('data_loaded_successfully', False):
        data_for_ui = st.session_state.working_data_full; available_years_in_data = st.session_state.available_years_in_data; project_names_list = st.session_state.project_names
        if not available_years_in_data: st.sidebar.warning("No usable year data in file.")
        else:
            st.markdown("## 2. Portfolio Settings")
            min_year = min(available_years_in_data); max_year_possible = max(available_years_in_data)
            years_max_slider = min(20, max(1, max_year_possible - min_year + 1))

            # --- Year Slider ---
            years_to_plan = st.slider(
                f"Years to Plan (Starting {min_year})", 1, years_max_slider,
                value=st.session_state.years_slider_sidebar, key='years_slider_sidebar'
            )
            start_year = min_year; end_year = start_year + years_to_plan - 1

            selected_years_range = list(range(start_year, end_year + 1)); actual_years_present_in_data = []
            for year in selected_years_range:
                price_col_display = f"price {year}"; vol_col_display = f"available volume {year}"
                if price_col_display in data_for_ui.columns and vol_col_display in data_for_ui.columns: actual_years_present_in_data.append(year)
            st.session_state.selected_years = actual_years_present_in_data
            if not st.session_state.selected_years:
                st.sidebar.error(f"No data for {start_year}-{end_year}. Adjust slider/check data."); st.session_state.actual_start_year = None; st.session_state.actual_end_year = None
            else:
                st.session_state.actual_start_year = min(st.session_state.selected_years); st.session_state.actual_end_year = max(st.session_state.selected_years)
                st.sidebar.markdown(f"Planning Horizon: **{st.session_state.actual_start_year} - {st.session_state.actual_end_year}**")

                # --- Constraint and Master Target (Syntax Corrected) ---
                st.session_state.constraint_type = st.radio("Constraint Type:", ('Volume', 'Budget'), index=['Volume', 'Budget'].index(st.session_state.get('constraint_type', 'Volume')), key='constraint_type_sidebar', horizontal=True)
                constraint_type = st.session_state.constraint_type
                master_target_value = st.session_state.get('master_target')

                if constraint_type == 'Volume':
                    default_val_vol = 1000
                    if master_target_value is not None:
                        try: default_val_vol = int(float(master_target_value))
                        except (ValueError, TypeError): pass
                    master_target = st.number_input("Target Volume (tonnes/year):", 0, step=100, value=default_val_vol, key='master_volume_sidebar')
                else: # Budget
                    default_val_bud = 100000.0
                    if master_target_value is not None:
                        try: default_val_bud = float(master_target_value)
                        except (ValueError, TypeError): pass
                    master_target = st.number_input("Target Budget (€/year):", 0.0, step=1000.0, value=default_val_bud, format="%.2f", key='master_budget_sidebar')

                st.session_state.master_target = master_target
                st.session_state.annual_targets = {year: master_target for year in st.session_state.selected_years}
                # --- End Corrected Block ---

                st.sidebar.markdown("### Allocation Goal")
                min_fulfill = st.sidebar.slider(f"Min. Target Fulfillment (%)", 50, 100, st.session_state.get('min_fulfillment_perc', 95), help=f"Attempt ≥ this % of target {constraint_type} via adjustment.", key='min_fulfill_perc_sidebar')
                st.session_state.min_fulfillment_perc = min_fulfill
                st.sidebar.markdown("### Removal Volume Transition")
                if 'reduction' in data_for_ui['project type'].unique(): st.sidebar.info("Note: Transition applies if 'Reduction' projects selected.")
                else: st.sidebar.info("Note: No 'Reduction' projects; transition inactive.")
                removal_help = f"Target % vol from Removals in final year ({st.session_state.actual_end_year}). Guides mix if 'Reduction' selected."
                rem_target_slider = st.sidebar.slider(f"Target Removal Vol % ({st.session_state.actual_end_year})", 0, 100, int(st.session_state.get('removal_target_end_year', 0.8)*100), help=removal_help, key='removal_perc_slider_sidebar')
                st.session_state.removal_target_end_year = rem_target_slider / 100.0
                st.session_state.transition_speed = st.sidebar.slider("Transition Speed", 1, 10, st.session_state.get('transition_speed', 5), help="Ramp-up speed (1=Slow, 10=Fast) if Reductions selected.", key='transition_speed_slider_sidebar')
                st.sidebar.markdown("### Removal Category Preference")
                rem_pref_val = st.sidebar.slider("Technical vs Natural Pref.", 1, 10, st.session_state.get('removal_preference_slider', 5), format="%d (1=Nat, 10=Tech)", help="Preference within removal target.", key='removal_pref_slider_sidebar')
                st.session_state['removal_preference_slider'] = rem_pref_val; tech_pref = (rem_pref_val - 1) / 9.0
                st.session_state.category_split = {'technical removal': tech_pref, 'natural removal': 1.0 - tech_pref}
                st.sidebar.markdown("## 3. Select Projects")
                if not project_names_list: st.sidebar.warning("No projects available.")
                else:
                    st.session_state.selected_projects = st.sidebar.multiselect("Select projects:", options=project_names_list, default=st.session_state.get('selected_projects', project_names_list), key='project_selector_sidebar')
                    if 'priority' in data_for_ui.columns:
                        boost_opts = [p for p in project_names_list if p in st.session_state.selected_projects]
                        if boost_opts:
                            curr_fav = st.session_state.get('favorite_projects_selection', [])
                            valid_def_fav = [f for f in curr_fav if f in boost_opts][:1]
                            st.session_state.favorite_projects_selection = st.sidebar.multiselect("Favorite Project (Boost):", options=boost_opts, default=valid_def_fav, key='favorite_selector_sidebar', max_selections=1, help="Boost priority for one project.")
                        else: st.sidebar.info("Select projects first to enable boost."); st.session_state.favorite_projects_selection = []
                    else: st.sidebar.info("No 'priority' column; boost disabled."); st.session_state.favorite_projects_selection = []

# ==================================
# Main Page Content
# ==================================
st.title("Carbon Portfolio Builder")

if not st.session_state.get('data_loaded_successfully', False):
    st.info("👋 Welcome! Please upload project data via the sidebar.")
elif not st.session_state.get('selected_projects'):
    st.warning("⚠️ Please select projects in the sidebar (Section 3).")
elif not st.session_state.get('selected_years'):
     st.warning("⚠️ No valid years for selected horizon. Adjust 'Years to Plan' (Section 2) or check data.")
else:
    required_keys = ['working_data_full', 'selected_projects', 'selected_years', 'actual_start_year', 'actual_end_year', 'constraint_type', 'annual_targets', 'removal_target_end_year', 'transition_speed', 'category_split', 'favorite_projects_selection', 'min_fulfillment_perc']
    keys_present = all(st.session_state.get(k) is not None for k in required_keys)

    # CORRECTED INDENTATION FOR THIS if/else BLOCK
    if keys_present:
        try:
            fav_proj = st.session_state.favorite_projects_selection[0] if st.session_state.favorite_projects_selection else None
            constraint = st.session_state.constraint_type
            min_chunk = 1
            if constraint == 'Budget': st.info(f"**Budget Mode:** Projects get budget via weighted priority. May get 0 vol if budget < cost of {min_chunk}t. Adjustment step may add later.")
            st.success(f"Goal: Attempt ≥ **{st.session_state.min_fulfillment_perc}%** of annual target {constraint}.")

            with st.spinner("Calculating portfolio..."):
                # The 'results' dictionary is detailed, 'summary' has yearly totals
                results, summary = allocate_portfolio(
                    project_data=st.session_state.working_data_full, selected_project_names=st.session_state.selected_projects,
                    selected_years=st.session_state.selected_years, start_year_portfolio=st.session_state.actual_start_year,
                    end_year_portfolio=st.session_state.actual_end_year, constraint_type=constraint,
                    annual_targets=st.session_state.annual_targets, removal_target_percent_end_year=st.session_state.removal_target_end_year,
                    transition_speed=st.session_state.transition_speed, category_split=st.session_state.category_split,
                    favorite_project=fav_proj, min_target_fulfillment_percent=st.session_state.min_fulfillment_perc / 100.0
                )

            # --- Display Results ---
            st.header("📈 Portfolio Allocation Results")

            # --- REMOVED Allocation Summary Table Display ---
            # st.markdown("#### Allocation Summary")
            # ... (code to display summary_disp dataframe was here) ...
            if summary.empty:
                 st.warning("Allocation calculation resulted in an empty summary.")


            # --- Viz & Combined Details ---
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

            # --- Display Combined Detailed Allocation Table (Pivoted: Year > Metric) with TOTAL Row --- ## INTEGRATED CHANGE ##
            st.markdown("#### 📄 Detailed Allocation (Pivoted: Year / Metric)")
            # Check if there are details OR a summary (summary is needed for totals)
            if not details_df.empty or not summary.empty:
                try:
                    pivot_final = pd.DataFrame() # Initialize empty DataFrame
                    years_present = st.session_state.selected_years # Default to selected years

                    if not details_df.empty:
                        # 1. Pivot with Metric on top level initially. NO fill_value here.
                        pivot_intermediate = pd.pivot_table(
                            details_df,
                            values=['volume', 'cost', 'price'],
                            index=['project name', 'type'],
                            columns='year',
                            aggfunc={'volume': 'sum', 'cost': 'sum', 'price': 'first'}
                        )

                        if not pivot_intermediate.empty:
                            # 2. Swap column levels
                            pivot_final = pivot_intermediate.swaplevel(0, 1, axis=1)

                            # 3. Sort columns
                            metric_order = ['volume', 'cost', 'price']
                            years_present = pivot_final.columns.get_level_values(0).unique() # Get actual years from pivot
                            final_multi_index = pd.MultiIndex.from_product(
                                [years_present, metric_order], names=['year', 'metric']
                            )
                            pivot_final = pivot_final.reindex(columns=final_multi_index)
                            pivot_final = pivot_final.sort_index(axis=1, level=0)
                            pivot_final.index.names = ['Project Name', 'Type'] # Set index names

                    # --- ADD TOTAL ROW USING SUMMARY DATA --- #
                    total_data = {}
                    summary_indexed = summary.set_index('Year') # Index summary by Year for easy lookup

                    for year in years_present: # Iterate through years actually present in data/columns
                        if year in summary_indexed.index:
                            # Extract values directly from the summary table for this year
                            vol = summary_indexed.loc[year, 'Allocated Volume']
                            cost = summary_indexed.loc[year, 'Allocated Cost']
                            avg_price = summary_indexed.loc[year, 'Avg. Price']
                        else:
                            # Default if year somehow not in summary (e.g., 0 target year)
                            vol, cost, avg_price = 0, 0.0, 0.0

                        # Populate the dictionary for the total row using summary values
                        total_data[(year, 'volume')] = vol
                        total_data[(year, 'cost')] = cost
                        total_data[(year, 'price')] = avg_price # Use avg price from summary

                    # Create the Total row DataFrame
                    total_row_index = pd.MultiIndex.from_tuples([('Total', 'All Types')], names=['Project Name', 'Type']) # Match index names
                    total_row_df = pd.DataFrame(total_data, index=total_row_index)

                    # Concatenate the total row
                    # If pivot_final was empty (no projects allocated), pivot_display becomes just the total row
                    pivot_display = pd.concat([pivot_final, total_row_df])

                    # Fill any NaNs remaining (e.g., price in project rows if not allocated)
                    pivot_display = pivot_display.fillna(0)
                    # --- END ADD TOTAL ROW --- #

                    # 5. Create the formatter dictionary dynamically
                    formatter = {}
                    for year, metric in pivot_display.columns:
                        if metric == 'volume': formatter[(year, metric)] = '{:,.0f} t'
                        elif metric == 'cost': formatter[(year, metric)] = '€{:,.2f}'
                        elif metric == 'price': formatter[(year, metric)] = '€{:,.2f}/t'

                    # 6. Display the final table including the total row
                    st.dataframe(
                        pivot_display.style.format(formatter, na_rep="-"),
                        use_container_width=True
                    )

                except Exception as e:
                    st.error(f"Could not create combined pivoted table: {e}")
                    st.error(f"Traceback: {traceback.format_exc()}")

            else: # Case where both details_df and summary are empty
                st.info("No allocation data or summary to display.")
            ## END INTEGRATED CHANGE ##


            # --- Visualization Section ---
            st.header("📊 Portfolio Analysis & Visualization")
            st.markdown("---")
            if details_df.empty: st.warning("No allocated projects. Cannot generate plots.")
            else:
                # --- Plotting code remains the same, uses details_df and summary ---
                colors = {'technical removal': '#66BB6A', 'natural removal': '#AED581', 'reduction': '#388E3C'}; default_c = '#BDBDBD'
                types_in_res = details_df['type'].unique()
                st.markdown("#### Portfolio Composition & Price Over Time")
                summary_plot = details_df.groupby(['year', 'type']).agg(volume=('volume', 'sum'), cost=('cost', 'sum')).reset_index()
                # Use summary df directly for avg price line if available
                price_sum = summary[['Year', 'Avg. Price']].rename(columns={'Year':'year', 'Avg. Price':'avg_price'})

                fig = make_subplots(specs=[[{"secondary_y": True}]]); type_order = ['reduction', 'natural removal', 'technical removal']
                y_met = 'volume' if constraint == 'Volume' else 'cost'; y_lab = 'Allocated Volume (t)' if constraint == 'Volume' else 'Allocated Cost (€)'
                y_fmt = '{:,.0f}' if constraint == 'Volume' else '€{:,.2f}'; y_hov = 'Volume' if constraint == 'Volume' else 'Cost'
                for t_name in type_order:
                    if t_name in types_in_res:
                        df_t = summary_plot[summary_plot['type'] == t_name]
                        if not df_t.empty and y_met in df_t.columns and df_t[y_met].sum() > 1e-6:
                            fig.add_trace(go.Bar(x=df_t['year'], y=df_t[y_met], name=t_name.capitalize(), marker_color=colors.get(t_name, default_c), hovertemplate=f'Year: %{{x}}<br>Type: {t_name.capitalize()}<br>{y_hov}: %{{y:{y_fmt}}}<extra></extra>'), secondary_y=False)
                # Use price_sum derived from summary df for the line plot
                if not price_sum.empty: fig.add_trace(go.Scatter(x=price_sum['year'], y=price_sum['avg_price'], name='Avg Price (€/t)', mode='lines+markers', marker=dict(symbol='circle', size=8), line=dict(color='#1B5E20', width=3), hovertemplate='Year: %{x}<br>Avg Price: €%{y:,.2f}<extra></extra>'), secondary_y=True)
                if not summary.empty and 'Actual Removal Vol %' in summary.columns: fig.add_trace(go.Scatter(x=summary['Year'], y=summary['Actual Removal Vol %'], name='Actual Removal Vol %', mode='lines+markers', line=dict(color='darkorange', dash='dash'), marker=dict(symbol='star'), hovertemplate='Year: %{x}<br>Actual Removal: %{y:.1f}%<extra></extra>'), secondary_y=True)
                y2_max = 105
                if not price_sum.empty and not price_sum['avg_price'].empty: y2_max = max(y2_max, price_sum['avg_price'].max() * 1.1)
                if not summary.empty and 'Actual Removal Vol %' in summary.columns and not summary['Actual Removal Vol %'].empty: y2_max = max(y2_max, summary['Actual Removal Vol %'].max() * 1.1)
                fig.update_layout(xaxis_title='Year', yaxis_title=y_lab, yaxis2_title='Avg Price (€/t) / Actual Removal %', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), barmode='stack', template="plotly_white", margin=dict(t=20, l=0, r=0, b=0), yaxis=dict(rangemode='tozero'), yaxis2=dict(rangemode='tozero', range=[0, y2_max]), hovermode="x unified")
                if st.session_state.selected_years: fig.update_xaxes(tickmode='array', tickvals=st.session_state.selected_years, dtick=1)
                st.plotly_chart(fig, use_container_width=True)

        # CORRECTED INDENTATION for except/else below
        except ValueError as e: st.error(f"Config/Allocation Error: {e}")
        except KeyError as e: st.error(f"Data Error: Missing key: '{e}'. Check CSV format/names & selections.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.error(f"Traceback: {traceback.format_exc()}")
    # This 'else' aligns with 'if keys_present:'
    else:
        st.error("Missing required settings in session state. Please reload data and configure settings.")

# --- Footer ---
try:
    # Use Zurich timezone as requested
    zurich_tz = pytz.timezone('Europe/Zurich')
    now_zurich = datetime.datetime.now(zurich_tz)
    st.caption(f"Generated: {now_zurich.strftime('%Y-%m-%d %H:%M:%S %Z')}")
except Exception as e: # Fallback if pytz not available or other error
    # st.error(f"Timezone error: {e}") # Optionally log the timezone error - removed for cleaner UI
    now_local = datetime.datetime.now()
    st.caption(f"Generated: {now_local.strftime('%Y-%m-%d %H:%M:%S')} (Timezone: Server Local)")
