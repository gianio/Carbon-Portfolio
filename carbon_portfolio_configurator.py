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
# Allocation Function (User Defined Logic - REVISED)
# ==================================
# --- Keep the allocate_portfolio function exactly as in the previous working version ---
# (No changes needed here for the requested modifications)
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
    REVISED:
    - Only considers projects listed in `selected_project_names`.
    - Renormalizes priorities within the selected projects for each type.
    - If no 'reduction' projects are selected, bypasses transition logic and allocates
      100% to selected removal projects based on `category_split`.
    - Removed dependency on `project_types_sliders`.

    (Args and Returns documentation omitted for brevity - same as before)
    """
    portfolio_details = {year: [] for year in selected_years}
    yearly_summary_list = []

    # --- 1. Filter Data to Selected Projects ONLY ---
    if not selected_project_names:
        return {}, pd.DataFrame(columns=[
            'Year', f'Target {constraint_type}', 'Allocated Volume',
            'Allocated Cost', 'Avg. Price', 'Actual Removal Vol %',
            'Target Removal Vol %'])
    project_data_selected = project_data[project_data['project name'].isin(selected_project_names)].copy()
    if project_data_selected.empty:
        return {}, pd.DataFrame(columns=[
            'Year', f'Target {constraint_type}', 'Allocated Volume',
            'Allocated Cost', 'Avg. Price', 'Actual Removal Vol %',
            'Target Removal Vol %'])

    all_project_types_in_selection = project_data_selected['project type'].unique()
    if len(all_project_types_in_selection) == 0:
         st.warning("Selected projects have no valid project types.")
         return {}, pd.DataFrame(columns=[
            'Year', f'Target {constraint_type}', 'Allocated Volume',
            'Allocated Cost', 'Avg. Price', 'Actual Removal Vol %',
            'Target Removal Vol %'])

    is_reduction_selected = 'reduction' in all_project_types_in_selection
    total_years_duration = end_year_portfolio - start_year_portfolio
    min_allocation_chunk = 1

    # --- Input Validation and Data Type Conversion ---
    required_cols = ['project name', 'project type', 'priority']
    price_cols_needed = []
    volume_cols_needed = []
    for year in selected_years:
        price_col = f"price {year}"; volume_col = f"available volume {year}"
        required_cols.extend([price_col, volume_col])
        price_cols_needed.append(price_col); volume_cols_needed.append(volume_col)

    missing_cols = [col for col in required_cols if col not in project_data_selected.columns]
    if missing_cols:
        missing_base = [c for c in ['project name', 'project type', 'priority'] if c not in project_data_selected.columns]
        missing_years_data = [c for c in required_cols if c not in project_data_selected.columns and c not in missing_base]
        err_msg = "Missing required columns in selected projects data: "
        if missing_base: err_msg += f"{', '.join(missing_base)}. "
        if missing_years_data: err_msg += f"Missing price/volume data for selected years (e.g., {missing_years_data[0]})."
        raise ValueError(err_msg)

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
        allocation_list_for_year = []

        summary_template = {
                'Year': year, f'Target {constraint_type}': yearly_target,
                'Allocated Volume': 0, 'Allocated Cost': 0, 'Avg. Price': 0,
                'Actual Removal Vol %': 0, 'Target Removal Vol %': 0 }

        if yearly_target <= 0:
            yearly_summary_list.append(summary_template)
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
                     if tech_selected: target_percentages['technical removal'] = 1.0
                     if nat_selected: target_percentages['natural removal'] = 1.0
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

        # --- Allocate within Types ---
        for project_type in all_project_types_in_selection:
            type_share = target_percentages.get(project_type, 0)
            if type_share <= 0: continue
            target_resource_for_type = yearly_target * type_share
            projects_in_type = project_data_selected[(project_data_selected['project type'] == project_type) & (project_data_selected[price_col] > 0) & (project_data_selected[volume_col] >= min_allocation_chunk)].copy()
            if projects_in_type.empty: continue

            # Priority Handling
            total_priority_in_type = projects_in_type['priority'].sum()
            if total_priority_in_type <= 0:
                num_projects = len(projects_in_type); equal_prio = 1.0 / num_projects if num_projects > 0 else 0
                projects_in_type['norm_prio_base'] = equal_prio
            else: projects_in_type['norm_prio_base'] = projects_in_type['priority'] / total_priority_in_type
            current_norm_priorities = projects_in_type.set_index('project name')['norm_prio_base'].to_dict()
            final_adjusted_priorities = current_norm_priorities.copy()

            # Favorite Project Boost
            if favorite_project and favorite_project in final_adjusted_priorities:
                fav_prio_norm = current_norm_priorities[favorite_project]; boost = 0.1; new_fav_prio = fav_prio_norm + boost
                other_project_names = [p for p in current_norm_priorities if p != favorite_project]
                sum_other_prio_norm = sum(current_norm_priorities[p] for p in other_project_names)
                temp_adjusted_prios = {favorite_project: new_fav_prio}
                if sum_other_prio_norm > 1e-9:
                     reduction_per_unit_prio = boost / sum_other_prio_norm
                     for p_name in other_project_names: temp_adjusted_prios[p_name] = max(0, current_norm_priorities[p_name] - (current_norm_priorities[p_name] * reduction_per_unit_prio))
                else:
                     for p_name in other_project_names: temp_adjusted_prios[p_name] = 0
                total_final_prio = sum(temp_adjusted_prios.values())
                if total_final_prio > 1e-9: final_adjusted_priorities = {p: prio / total_final_prio for p, prio in temp_adjusted_prios.items()}
                else: num_projs = len(temp_adjusted_prios); final_adjusted_priorities = {p: 1.0/num_projs if num_projs>0 else 0 for p in temp_adjusted_prios}
                if favorite_project not in final_adjusted_priorities and len(final_adjusted_priorities) == 0 and len(temp_adjusted_prios) == 1: final_adjusted_priorities = {favorite_project: 1.0}

            # Budget Weight Calculation
            total_weight_for_type = 0; project_weights = {}
            if constraint_type == 'Budget':
                for _, project_row in projects_in_type.iterrows():
                    proj_name = project_row['project name']; p_prio_final = final_adjusted_priorities.get(proj_name, 0); p_price = project_row[price_col]
                    current_weight = p_prio_final * p_price; project_weights[proj_name] = current_weight; total_weight_for_type += current_weight

            # Resource Allocation Loop
            for _, project_row in projects_in_type.iterrows():
                proj_name = project_row['project name']; proj_prio_final = final_adjusted_priorities.get(proj_name, 0)
                available_vol = project_row[volume_col]; price = project_row[price_col]
                allocated_volume = 0; allocated_cost = 0
                if proj_prio_final <= 0 or price <= 0: continue

                if constraint_type == 'Volume':
                    project_target_volume = target_resource_for_type * proj_prio_final
                    allocated_volume = min(project_target_volume, available_vol)
                elif constraint_type == 'Budget':
                    if total_weight_for_type > 1e-9:
                        normalized_weight = project_weights.get(proj_name, 0) / total_weight_for_type
                        project_target_budget = target_resource_for_type * normalized_weight
                        project_target_volume = project_target_budget / price
                        allocated_volume = min(project_target_volume, available_vol)
                    else: allocated_volume = 0

                allocated_volume = int(max(0, math.floor(allocated_volume)))
                if allocated_volume >= min_allocation_chunk:
                    allocated_cost = allocated_volume * price
                    allocation_list_for_year.append({'project name': proj_name, 'type': project_type, 'allocated_volume': allocated_volume, 'allocated_cost': allocated_cost, 'price_used': price, 'priority_applied': proj_prio_final })
                    year_total_allocated_vol += allocated_volume; year_total_allocated_cost += allocated_cost

        # --- Store results ---
        portfolio_details[year] = allocation_list_for_year
        summary_template['Allocated Volume'] = year_total_allocated_vol
        summary_template['Allocated Cost'] = year_total_allocated_cost
        summary_template['Avg. Price'] = (year_total_allocated_cost / year_total_allocated_vol) if year_total_allocated_vol > 0 else 0
        removal_vol = sum(p['allocated_volume'] for p in allocation_list_for_year if p['type'] != 'reduction')
        summary_template['Actual Removal Vol %'] = (removal_vol / year_total_allocated_vol * 100) if year_total_allocated_vol > 0 else 0
        yearly_summary_list.append(summary_template)

    yearly_summary_df = pd.DataFrame(yearly_summary_list)

    # Budget Check
    if constraint_type == 'Budget':
        summary_check = yearly_summary_df.copy()
        master_target_val = list(annual_targets.values())[0] if annual_targets else 0
        summary_check['Target Budget'] = master_target_val
        overbudget_years = summary_check[summary_check['Allocated Cost'] > summary_check['Target Budget'] * 1.001]
        if not overbudget_years.empty: print(f"WARNING: Budget potentially exceeded in years: {overbudget_years['Year'].tolist()}")

    return portfolio_details, yearly_summary_df


# ==================================
# Streamlit App Layout & Logic (SIDEBAR REVERTED)
# ==================================

# --- Sidebar (Sections 1, 2, 3) ---
with st.sidebar:
    st.markdown("## 1. Load Data")
    df_upload = st.file_uploader("Upload Project Data CSV", type="csv", key="uploader_sidebar") # Use sidebar key

    # Initialize session state
    # ... (session state initialization remains the same) ...
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
    if 'data_loaded_successfully' not in st.session_state: st.session_state.data_loaded_successfully = False # Flag for loading
    if 'years_slider_sidebar' not in st.session_state: st.session_state.years_slider_sidebar = 5 # Default slider value


    if df_upload:
        try:
            # --- Data Loading and Preparation (same as before) ---
            @st.cache_data
            def load_and_prepare_data(uploaded_file):
                # ... (loading logic identical to previous version) ...
                data = pd.read_csv(uploaded_file)
                data.columns = data.columns.str.lower().str.strip().str.replace(' ', '_')
                core_cols_std = ['project_name', 'project_type', 'priority']
                missing_essential = [col for col in core_cols_std if col not in data.columns]
                if missing_essential: return None, f"Missing essential columns: {', '.join(missing_essential)}", [], [], []
                numeric_prefixes_std = ['price_', 'available_volume_']
                cols_to_convert = ['priority']
                available_years = set()
                for col in data.columns:
                    for prefix in numeric_prefixes_std:
                        year_part = col[len(prefix):]
                        if col.startswith(prefix) and year_part.isdigit():
                            cols_to_convert.append(col); available_years.add(int(year_part)); break
                for col in list(set(cols_to_convert)):
                    if col in data.columns: data[col] = pd.to_numeric(data[col], errors='coerce')
                data['priority'] = data['priority'].fillna(0)
                for col in data.columns:
                     if col.startswith('available_volume_'): data[col] = data[col].fillna(0).apply(lambda x: max(0, int(x)) if pd.notna(x) else 0).clip(lower=0)
                     if col.startswith('price_'): data[col] = data[col].fillna(0.0).apply(lambda x: max(0.0, float(x)) if pd.notna(x) else 0.0).clip(lower=0)
                available_years = sorted(list(available_years))
                if not available_years: return None, "No valid year data columns found (e.g., 'price_YYYY').", [], [], []
                invalid_types_found = []
                if 'project_type' in data.columns:
                    data['project_type'] = data['project_type'].str.lower().str.strip()
                    valid_types = ['reduction', 'technical removal', 'natural removal']
                    invalid_types_df = data[~data['project_type'].isin(valid_types)]
                    if not invalid_types_df.empty:
                         invalid_types_found = invalid_types_df['project_type'].unique().tolist()
                         data = data[data['project_type'].isin(valid_types)]
                else: return None, "Missing 'project_type' column.", available_years, [], []
                column_mapping_to_display = {col: col.replace('_', ' ') for col in data.columns}
                data.rename(columns=column_mapping_to_display, inplace=True)
                project_names_list = sorted(data['project name'].unique().tolist()) if 'project name' in data.columns else []
                return data, None, available_years, project_names_list, invalid_types_found

            data, error_msg, available_years_in_data, project_names_list, invalid_types_found = load_and_prepare_data(df_upload)

            if invalid_types_found: st.sidebar.warning(f"Invalid project types ignored: {', '.join(invalid_types_found)}")
            if error_msg:
                st.sidebar.error(error_msg)
                st.session_state.data_loaded_successfully = False
                st.session_state.working_data_full = None; st.session_state.project_names = []; st.session_state.available_years_in_data = []
            else:
                st.session_state.project_names = project_names_list
                st.session_state.available_years_in_data = available_years_in_data
                st.session_state.working_data_full = data
                st.session_state.data_loaded_successfully = True
                st.sidebar.success("Data loaded!")
                # Initialize selected projects if not already set or if data changed
                current_selection = st.session_state.get('selected_projects', [])
                valid_current_selection = [p for p in current_selection if p in project_names_list]
                if not valid_current_selection : # If selection empty or invalid relative to new data
                     st.session_state.selected_projects = project_names_list

        except Exception as e:
            st.sidebar.error(f"Error processing file: {e}")
            st.session_state.data_loaded_successfully = False
            st.session_state.working_data_full = None

    # --- Sections 2 & 3 Now inside Sidebar, dependent on successful load ---
    if st.session_state.get('data_loaded_successfully', False):
        data = st.session_state.working_data_full
        available_years_in_data = st.session_state.available_years_in_data
        project_names_list = st.session_state.project_names

        # --- Section 2: Portfolio Settings ---
        st.markdown("## 2. Portfolio Settings")
        min_year = min(available_years_in_data); max_year_possible = max(available_years_in_data)
        years_max_slider = min(20, max_year_possible - min_year + 1); years_max_slider = max(1, years_max_slider)

        # Persist slider value using sidebar key
        years_to_plan = st.slider(f"Years to Plan (from {min_year})", 1, years_max_slider, st.session_state.years_slider_sidebar, key='years_slider_sidebar')

        start_year = min_year; end_year = start_year + years_to_plan - 1
        selected_years_range = list(range(start_year, end_year + 1))
        actual_years_present = []
        for year in selected_years_range:
            price_col = f"price {year}"; vol_col = f"available volume {year}"
            if price_col in data.columns and vol_col in data.columns:
                actual_years_present.append(year)
        st.session_state.selected_years = actual_years_present

        if not st.session_state.selected_years:
            st.sidebar.error(f"No price/volume data found for years {start_year}-{end_year}.")
        else:
            st.session_state.actual_start_year = min(st.session_state.selected_years)
            st.session_state.actual_end_year = max(st.session_state.selected_years)
            st.sidebar.markdown(f"Planning Horizon: **{st.session_state.actual_start_year} - {st.session_state.actual_end_year}**")

            # --- Constraint and Master Target ---
            st.session_state.constraint_type = st.radio(
                "Constraint Type:", ('Volume', 'Budget'),
                index=['Volume', 'Budget'].index(st.session_state.get('constraint_type', 'Volume')),
                key='constraint_type_sidebar', horizontal=True # Use sidebar key
                )
            constraint_type = st.session_state.constraint_type

            master_target_value_from_state = st.session_state.get('master_target')
            if constraint_type == 'Volume':
                default_val_vol = 1000
                if master_target_value_from_state is not None:
                    try: default_val_vol = int(float(master_target_value_from_state))
                    except (ValueError, TypeError): default_val_vol = 1000
                master_target = st.number_input("Target Volume (per year):", min_value=0, step=100, value=default_val_vol, key='master_volume_sidebar') # Use sidebar key
            else: # Budget
                default_val_bud = 100000.0
                if master_target_value_from_state is not None:
                    try: default_val_bud = float(master_target_value_from_state)
                    except (ValueError, TypeError): default_val_bud = 100000.0
                master_target = st.number_input("Target Budget (€ per year):", min_value=0.0, step=1000.0, value=default_val_bud, format="%.2f", key='master_budget_sidebar') # Use sidebar key

            st.session_state.master_target = master_target
            st.session_state.annual_targets = {year: master_target for year in st.session_state.selected_years}

            # --- Removal Transition Settings ---
            st.sidebar.markdown("### Removal Volume Transition") # Use sidebar context
            st.sidebar.info("Note: Transition settings apply only if 'Reduction' projects are selected.")
            removal_help_text = (
                f"Target % of *volume* from Removals in the final year ({st.session_state.actual_end_year}). Starts at 10%. "
                "If 'Reduction' projects selected, this guides allocation. "
                "In 'Budget' mode, actual volume % may differ due to prices."
            )
            removal_target_percent_slider = st.sidebar.slider( # Use sidebar context
                f"Target Removal Vol % for Year {st.session_state.actual_end_year}", 0, 100,
                int(st.session_state.get('removal_target_end_year', 0.8)*100),
                help=removal_help_text, key='removal_perc_slider_sidebar' # Use sidebar key
                )
            st.session_state.removal_target_end_year = removal_target_percent_slider / 100.0

            st.session_state.transition_speed = st.sidebar.slider( # Use sidebar context
                "Transition Speed", 1, 10, st.session_state.get('transition_speed', 5),
                help="Speed of ramping up removal % (1=Slow, 10=Fast), if Reductions selected.",
                key='transition_speed_slider_sidebar' # Use sidebar key
                )

            # --- Technical/Natural Split ---
            st.sidebar.markdown("### Removal Category Preference") # Use sidebar context
            removal_preference_val = st.sidebar.slider( # Use sidebar context
                "Technical Removal Preference", 1, 10,
                st.session_state.get('removal_preference_slider', 5),
                help="Focus within removal target (1=More Natural, 10=More Technical)",
                key='removal_pref_slider_sidebar' # Use sidebar key
                )
            st.session_state['removal_preference_slider'] = removal_preference_val
            tech_pref_norm = (removal_preference_val - 1) / 9.0
            st.session_state.category_split = {'technical removal': tech_pref_norm, 'natural removal': 1.0 - tech_pref_norm}

            # --- Section 3: Project Selection ---
            st.sidebar.markdown("## 3. Select Projects") # Use sidebar context
            st.session_state.selected_projects = st.sidebar.multiselect( # Use sidebar context
                "Select projects for portfolio:",
                project_names_list,
                default=st.session_state.get('selected_projects', project_names_list),
                key='project_selector_sidebar' # Use sidebar key
                )

            available_for_boost = [p for p in project_names_list if p in st.session_state.selected_projects]
            current_favorites = st.session_state.get('favorite_projects_selection', [])
            valid_default_favorites = [fav for fav in current_favorites if fav in available_for_boost]

            if 'priority' in data.columns:
                if available_for_boost:
                    st.session_state.favorite_projects_selection = st.sidebar.multiselect( # Use sidebar context
                        "Select Favorite Project (Boosts Priority):",
                        available_for_boost,
                        default=valid_default_favorites,
                        key='favorite_selector_sidebar' # Use sidebar key
                        )
                else:
                     st.sidebar.info("Select projects above to enable favorite boosting.")
                     st.session_state.favorite_projects_selection = []
            else:
                 st.sidebar.info("No 'priority' column found; boosting disabled.")
                 st.session_state.favorite_projects_selection = []

# ==================================
# Main Page Content (RESULTS ONLY + DYNAMIC Y-AXIS)
# ==================================
st.title("Carbon Portfolio Builder")

# Check if data is loaded AND projects are selected before attempting calculation
if not st.session_state.get('data_loaded_successfully', False):
    st.info("👋 Welcome! Please upload data via the sidebar to begin.")
elif not st.session_state.get('selected_projects'):
    st.warning("⚠️ Please select projects in the sidebar to calculate the portfolio.")
elif not st.session_state.get('selected_years'):
     st.warning("⚠️ No valid years selected based on slider and data availability. Adjust 'Years to Plan' in the sidebar.")
else:
    # Check if all required session state keys for calculation are present and valid
    required_state_keys = [
        'working_data_full', 'selected_projects', 'selected_years',
        'annual_targets', 'constraint_type',
        'removal_target_end_year', 'transition_speed', 'category_split',
        'favorite_projects_selection', 'actual_start_year', 'actual_end_year'
    ]
    required_state_keys_present = all(st.session_state.get(key) is not None for key in required_state_keys)
    valid_years_selected = bool(st.session_state.get('selected_years'))

    if required_state_keys_present and valid_years_selected:
        try:
            favorite_project_name = st.session_state.favorite_projects_selection[0] if st.session_state.favorite_projects_selection else None
            current_constraint_type = st.session_state.constraint_type # Get constraint type for plot labels

            # Add specific note for budget mode about project inclusion
            if current_constraint_type == 'Budget':
                st.info(
                    "**Budget Mode Note:** Projects are allocated budget based on priority and price. "
                    "Selected projects might receive 0 volume if their allocated budget share is too small "
                    "to purchase at least 1 tonne, especially if they have high prices or low relative priority."
                )

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

            # --- Display results ---
            st.header("📈 Portfolio Allocation Results")
            st.markdown("#### Allocation Summary")
            if not summary_df.empty:
                format_dict = {
                    f'Target {current_constraint_type}': '€{:,.2f}' if current_constraint_type == 'Budget' else '{:,.0f}',
                    'Allocated Volume': '{:,.0f}', 'Allocated Cost': '€{:,.2f}', 'Avg. Price': '€{:,.2f}',
                    'Actual Removal Vol %': '{:.1f}%', 'Target Removal Vol %': '{:.1f}%'
                }
                st.dataframe(summary_df.style.format(format_dict), hide_index=True, use_container_width=True)
                if current_constraint_type == 'Budget':
                    st.caption(
                        "Note: In 'Budget' mode, the 'Target Removal Vol %' reflects the intended budget split trend. "
                        "The 'Actual Removal Vol %' achieved depends on relative project prices."
                    )
            else: st.warning("Allocation resulted in an empty summary.")

            # --- Detailed Allocation Tabs ---
            if portfolio_results and any(portfolio_results.values()):
                st.markdown("#### 📄 Detailed Allocation per Year")
                # ... (Detailed tabs display logic remains the same) ...
                years_with_allocations = [yr for yr in st.session_state.selected_years if yr in portfolio_results and portfolio_results[yr]]
                if years_with_allocations:
                    tab_list = st.tabs([str(year) for year in years_with_allocations])
                    for i, year_tab in enumerate(years_with_allocations):
                        with tab_list[i]:
                            year_data_list = portfolio_results[year_tab]
                            if year_data_list:
                                year_df = pd.DataFrame(year_data_list).sort_values(by='project name')
                                display_df = year_df[['project name', 'type', 'allocated_volume', 'allocated_cost', 'price_used', 'priority_applied']]
                                st.dataframe(display_df.style.format({
                                    'allocated_volume': '{:,.0f}', 'allocated_cost': '€{:,.2f}',
                                    'price_used': '€{:,.2f}', 'priority_applied': '{:.4f}'
                                    }), hide_index=True, use_container_width=True)
                else: st.info("No projects allocated in any selected year.")

            # --- Visualization ---
            st.header("📊 Portfolio Analysis & Visualization")
            st.markdown("---")
            type_color_map = {'technical removal': '#66BB6A', 'natural removal': '#AED581', 'reduction': '#388E3C'}
            default_color = '#BDBDBD'
            project_color_scale = px.colors.sequential.Greens

            portfolio_data_list_viz = []
            if portfolio_results:
                for year_viz, projects_list in portfolio_results.items():
                    if projects_list:
                        for proj_info in projects_list:
                             if proj_info.get('allocated_volume', 0) > 0 or proj_info.get('allocated_cost', 0) > 0: # Include if either vol or cost > 0
                                portfolio_data_list_viz.append({
                                    'year': year_viz, 'project name': proj_info['project name'], 'type': proj_info['type'],
                                    'volume': proj_info['allocated_volume'], 'price': proj_info['price_used'], 'cost': proj_info['allocated_cost']
                                })

            if not portfolio_data_list_viz:
                 st.warning("No projects with positive allocation found. Cannot generate plots.")
            else:
                portfolio_df_viz = pd.DataFrame(portfolio_data_list_viz)
                valid_types_in_results = portfolio_df_viz['type'].unique()

                # --- Composition & Price Plot (DYNAMIC Y-AXIS) ---
                st.markdown("#### Portfolio Composition & Price Over Time")
                summary_plot_df = portfolio_df_viz.groupby(['year', 'type']).agg(volume=('volume', 'sum'), cost=('cost', 'sum')).reset_index()
                price_summary = summary_plot_df.groupby('year').agg(total_volume=('volume', 'sum'), total_cost=('cost', 'sum')).reset_index()
                price_summary['avg_price'] = price_summary.apply(lambda row: row['total_cost'] / row['total_volume'] if row['total_volume'] > 0 else 0, axis=1)

                fig_composition = make_subplots(specs=[[{"secondary_y": True}]])
                plot_type_order = ['reduction', 'natural removal', 'technical removal']

                # Determine primary Y-axis data and label based on constraint type
                y_axis_metric = 'volume' if current_constraint_type == 'Volume' else 'cost'
                y_axis_label = 'Volume (tonnes)' if current_constraint_type == 'Volume' else 'Allocated Cost (€)'
                y_axis_format = '{:,.0f}' if current_constraint_type == 'Volume' else '€{:,.2f}'
                y_axis_hover_label = 'Volume' if current_constraint_type == 'Volume' else 'Cost'


                for type_name in plot_type_order:
                    if type_name in valid_types_in_results:
                        df_type = summary_plot_df[summary_plot_df['type'] == type_name]
                        if not df_type.empty and y_axis_metric in df_type.columns: # Ensure metric exists
                            fig_composition.add_trace(
                                go.Bar(x=df_type['year'], y=df_type[y_axis_metric], name=type_name.capitalize(),
                                       marker_color=type_color_map.get(type_name, default_color),
                                       # Update hovertemplate dynamically
                                       hovertemplate=f'Year: %{{x}}<br>Type: {type_name.capitalize()}<br>{y_axis_hover_label}: %{{y:{y_axis_format}}}<extra></extra>'
                                       ), secondary_y=False)

                # Add secondary axes (Price and Removal %) - these remain the same
                if not price_summary.empty:
                    fig_composition.add_trace(go.Scatter(x=price_summary['year'], y=price_summary['avg_price'], name='Avg Price (€/t)', mode='lines+markers', marker=dict(symbol='circle', size=8), line=dict(color='#1B5E20', width=3), hovertemplate='Year: %{x}<br>Avg Price: €%{y:,.2f}<extra></extra>'), secondary_y=True)
                if not summary_df.empty and 'Actual Removal Vol %' in summary_df.columns:
                    fig_composition.add_trace(go.Scatter(x=summary_df['Year'], y=summary_df['Actual Removal Vol %'], name='Actual Removal Vol %', mode='lines+markers', line=dict(color='darkorange', dash='dash'), marker=dict(symbol='star'), hovertemplate='Year: %{x}<br>Actual Removal: %{y:.1f}%<extra></extra>'), secondary_y=True)

                y2_max = 105
                if not price_summary.empty and not price_summary['avg_price'].empty: y2_max = max(y2_max, price_summary['avg_price'].max() * 1.1)

                fig_composition.update_layout(
                    xaxis_title='Year',
                    yaxis_title=y_axis_label, # Dynamic Y-axis label
                    yaxis2_title='Avg Price (€/t) / Actual Removal %',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    barmode='stack', template="plotly_white", margin=dict(t=20, l=0, r=0, b=0),
                    yaxis=dict(rangemode='tozero'), yaxis2=dict(rangemode='tozero', range=[0, y2_max]), hovermode="x unified"
                    )
                fig_composition.update_xaxes(tickmode='array', tickvals=st.session_state.selected_years, dtick=1)
                st.plotly_chart(fig_composition, use_container_width=True)

                # --- Sunburst Chart (remains based on Volume) ---
                st.markdown("#### Total Allocation Overview (Volume)")
                if not portfolio_df_viz.empty:
                    # Use Volume for Sunburst regardless of constraint type for consistency in this view
                    df_summed_sunburst = portfolio_df_viz.groupby(['type', 'project name']).agg(volume=('volume', 'sum')).reset_index()
                    df_summed_sunburst = df_summed_sunburst[df_summed_sunburst['volume'] > 0]
                    if not df_summed_sunburst.empty:
                        df_summed_sunburst['display_name'] = df_summed_sunburst['project name']
                        fig_sunburst = px.sunburst(df_summed_sunburst, path=['type', 'display_name'], values='volume',
                                                   color='type', color_discrete_map=type_color_map, title=None, branchvalues="total")
                        fig_sunburst.update_traces(textinfo='label+percent parent', insidetextorientation='radial',
                                                   hovertemplate='<b>%{label}</b><br>Total Volume: %{value:,.0f}<br>% of Parent: %{percentParent:.1%}<extra></extra>')
                        fig_sunburst.update_layout(margin=dict(t=10, l=0, r=0, b=0))
                        st.plotly_chart(fig_sunburst, use_container_width=True)

                # --- Detailed Project Allocation Plot (DYNAMIC Y-AXIS) ---
                st.markdown(f"#### Detailed Project Allocation Over Time ({y_axis_hover_label})")
                if not portfolio_df_viz.empty:
                    fig_grouped_projects = go.Figure()
                    unique_projects = sorted(portfolio_df_viz['project name'].unique())
                    project_color_map = {proj: project_color_scale[i % len(project_color_scale)] for i, proj in enumerate(unique_projects)}

                    for type_name in plot_type_order:
                        if type_name in valid_types_in_results:
                            type_projects_df = portfolio_df_viz[portfolio_df_viz['type'] == type_name]
                            projects_in_type = sorted(type_projects_df['project name'].unique())
                            for project in projects_in_type:
                                project_data = type_projects_df[type_projects_df['project name'] == project]
                                if not project_data.empty and y_axis_metric in project_data.columns: # Ensure metric exists
                                     # Check if the value for the metric is positive before plotting
                                     valid_rows = project_data[project_data[y_axis_metric] > 1e-6] # Use small tolerance instead of 0
                                     if not valid_rows.empty:
                                        fig_grouped_projects.add_trace(
                                            go.Bar(x=valid_rows['year'], y=valid_rows[y_axis_metric], name=project,
                                                   marker_color=project_color_map.get(project, default_color),
                                                   legendgroup=type_name, legendgrouptitle_text=type_name.capitalize(),
                                                   # Update hovertemplate dynamically
                                                   hovertemplate = f'Year: %{{x}}<br>Project: {project}<br>{y_axis_hover_label}: %{{y:{y_axis_format}}}<extra></extra>'
                                                   ))

                    fig_grouped_projects.update_layout(
                        xaxis_title='Year',
                        yaxis_title=y_axis_label, # Dynamic Y-axis label
                        legend_title='Projects by Type',
                        legend=dict(tracegroupgap=10, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        barmode='stack', template="plotly_white", yaxis=dict(rangemode='tozero'),
                        margin=dict(t=20, l=0, r=0, b=0), # Small top margin
                        xaxis=dict(tickmode='array', tickvals=st.session_state.selected_years, dtick=1)
                        )
                    st.plotly_chart(fig_grouped_projects, use_container_width=True)


                # --- Raw Data Table ---
                if not portfolio_df_viz.empty:
                    with st.expander("Show Detailed Allocation Data Table (All Years)"):
                        display_portfolio_df = portfolio_df_viz.copy()
                        display_portfolio_df['volume'] = display_portfolio_df['volume'].map('{:,.0f}'.format)
                        display_portfolio_df['price'] = display_portfolio_df['price'].map('€{:,.2f}'.format)
                        display_portfolio_df['cost'] = display_portfolio_df['cost'].map('€{:,.2f}'.format)
                        st.dataframe(display_portfolio_df[['year', 'project name', 'type', 'volume', 'price', 'cost']].sort_values(by=['year', 'project name']), hide_index=True, use_container_width=True)

        except ValueError as e:
            st.error(f"Configuration or Allocation Error: {e}")
        except KeyError as e:
            st.error(f"Data Error: Missing expected column or key: {e}. Check CSV format/selections.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            # st.exception(e) # Uncomment for detailed traceback during debugging

    # Simple footer or date check
    current_date = datetime.date.today()
    st.caption(f"Analysis generated on: {current_date.strftime('%Y-%m-%d')}")
