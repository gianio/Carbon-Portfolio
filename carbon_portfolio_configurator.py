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
import unicodedata # For Unicode normalization
# import re # Only needed if using re.sub for more aggressive underscore replacement

# ==================================
# Refactored Data Loading Function (with aggressive normalization)
# ==================================
@st.cache_data
def load_and_prepare_data_simplified(uploaded_file):
    """
    Loads data from an uploaded CSV file, prepares and validates it.
    Includes aggressive Unicode NFKC normalization for column headers.
    """
    try:
        data = pd.read_csv(uploaded_file, encoding='utf-8-sig', skipinitialspace=True)
        
        standardized_columns = []
        for col in data.columns:
            col_str = str(col)
            # 1. Unicode Normalization (NFKC form)
            normalized_col = unicodedata.normalize('NFKC', col_str)
            # 2. Lowercase
            lower_col = normalized_col.lower()
            # 3. Strip leading/trailing whitespace (again, as normalization might change things)
            stripped_col = lower_col.strip()
            # 4. Replace internal spaces (now likely standard spaces) with underscores
            final_col = stripped_col.replace(' ', '_')
            # Optional: Consolidate multiple underscores if replace(' ', '_') could create them
            # import re
            # final_col = re.sub(r'_+', '_', final_col) 
            standardized_columns.append(final_col)
        data.columns = standardized_columns

    except Exception as read_error:
        return None, f"Error reading or initially processing CSV file: {read_error}. Ensure it's a valid CSV.", [], [], []

    core_cols_std = ['project_name', 'project_type', 'priority'] # These are already in a "clean" state
    
    data_columns_set = set(data.columns) # data.columns are now aggressively standardized
    missing_essential = [col for col in core_cols_std if col not in data_columns_set]

    if missing_essential:
        data_columns_repr_list = [repr(col_name) for col_name in data.columns]
        core_cols_repr_list = [repr(col_name) for col_name in core_cols_std]

        err_msg = (
            f"CSV missing essential columns after thorough standardization (including Unicode NFKC normalization). \n"
            f"1. Expected standardized names (should be plain ASCII): {core_cols_std}\n"
            f"   Representation (repr) of expected names: {core_cols_repr_list}\n"
            f"2. Thoroughly standardized columns found in uploaded file: {list(data.columns)}\n"
            f"   Representation (repr) of these found columns: {data_columns_repr_list}\n"
            f"3. Essential columns that were NOT found in the standardized list from data: {missing_essential}\n"
            f"This indicates a persistent, unexpected mismatch. Please verify the CSV headers are simple, plain text (ideally ASCII/UTF-8) and comma-separated. "
            f"If the 'repr' for an expected column and a found column look identical but are still mismatched, there might be a very subtle encoding or invisible character issue not fully resolved by NFKC, or an environment-specific string comparison problem."
        )
        return None, err_msg, [], [], []

    available_years = set()
    year_data_cols_found_std = []
    numeric_prefixes_std = ['price_', 'available_volume_']
    project_names_list_intermediate = [] 
    if 'project_name' in data.columns: # Check using the already standardized name
        try:
            project_names_list_intermediate = sorted(data['project_name'].astype(str).unique().tolist())
        except Exception: 
             project_names_list_intermediate = []


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
        return None, "Core columns found, but no valid year columns (e.g., 'price_YYYY', 'available_volume_YYYY') were detected after processing headers. The application requires these year-specific columns to function.", [], project_names_list_intermediate, []

    available_years = sorted(list(available_years))
    data['priority'] = pd.to_numeric(data['priority'], errors='coerce').fillna(0)

    for year_val_proc in available_years:
        price_col_std_proc = f"price_{year_val_proc}"
        volume_col_std_proc = f"available_volume_{year_val_proc}"

        if price_col_std_proc in year_data_cols_found_std and price_col_std_proc in data.columns:
            data[price_col_std_proc] = pd.to_numeric(data[price_col_std_proc], errors='coerce')
            data[price_col_std_proc] = data[price_col_std_proc].fillna(0.0).clip(lower=0.0)
        
        if volume_col_std_proc in year_data_cols_found_std and volume_col_std_proc in data.columns:
            data[volume_col_std_proc] = pd.to_numeric(data[volume_col_std_proc], errors='coerce')
            data[volume_col_std_proc] = data[volume_col_std_proc].fillna(0).round().astype(int).clip(lower=0)

    invalid_types_found_list = []
    # project_type column existence is already guaranteed by core_cols_std check
    data['project_type'] = data['project_type'].astype(str).str.lower().str.strip() # Already NFKC normalized, lowercased, stripped. This is redundant but harmless.
    valid_types = ['reduction', 'technical removal', 'natural removal']
    invalid_rows_mask = ~data['project_type'].isin(valid_types)
    if invalid_rows_mask.any():
        invalid_types_found_list = data.loc[invalid_rows_mask, 'project_type'].unique().tolist()
        data = data[~invalid_rows_mask].copy()
        if data.empty:
            return None, f"All rows had invalid project types or no valid project types left after filtering. Valid types: {', '.join(valid_types)}. Found: {', '.join(invalid_types_found_list)}", available_years, [], invalid_types_found_list
    
    cols_to_keep_std = core_cols_std[:]
    optional_cols_std = ['description', 'project_link']
    for col_opt in optional_cols_std:
        if col_opt in data.columns: # data.columns are already fully standardized
            cols_to_keep_std.append(col_opt)
    cols_to_keep_std.extend([col_yr for col_yr in year_data_cols_found_std if col_yr in data.columns])
    
    final_cols_to_keep = [col_keep for col_keep in list(set(cols_to_keep_std)) if col_keep in data.columns]
    data = data[final_cols_to_keep].copy()

    rename_map_to_desired = {
        'project_name': 'project name', # Keys are clean, standardized names
        'project_type': 'project type',
        'priority': 'priority', 
        'description': 'Description',
        'project_link': 'Project Link'
    }
    for year_col_std_rename in year_data_cols_found_std: # These are like 'price_2023'
        if year_col_std_rename in data.columns: 
             rename_map_to_desired[year_col_std_rename] = year_col_std_rename.replace('_', ' ') # 'price_2023' -> 'price 2023'

    data.rename(columns=rename_map_to_desired, inplace=True)

    final_project_names_list = []
    if 'project name' in data.columns: # Check the finally renamed column
        final_project_names_list = sorted(data['project name'].astype(str).unique().tolist())

    return data, None



# ==================================
# Configuration & Theming
# ==================================
st.set_page_config(layout="wide")
# --- Combined CSS ---
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
    }
    .metric-box b { /* Style for the label */
        display: block;
        margin-bottom: 5px;
        color: #1B5E20; /* Dark green label */
        font-size: 0.6em; /* Adjust label font size relative to value */
        font-weight: bold;
    }

    /* Experimental: Increase font size in dataframes */
    .stDataFrame table td, .stDataFrame table th {
        font-size: 115%; /* Adjusted font size */
    }

    /* Styling for the download button */
    div[data-testid="stDownloadButton"] > button {
        background-color: #8ca734 !important; 
        color: white !important;
        border: none !important; 
        padding: 0.8em 1.5em !important; 
        width: auto; 
        font-size: 1.1em !important; 
        font-weight: bold;
        border-radius: 5px; 
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: #6b8e23 !important; 
        color: white !important;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Define Color Map for Charts (Green Theme)
type_color_map = {
    'technical removal': '#66BB6A', 'natural removal': '#AED581', 'reduction': '#388E3C'
}
default_color = '#BDBDBD' 


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
    portfolio_details = {year_alloc: [] for year_alloc in selected_years} # Renamed year
    yearly_summary_list = []

    if not selected_project_names:
        st.warning("No projects selected for allocation.")
        return {}, pd.DataFrame(columns=['Year', f'Target {constraint_type}', 'Allocated Volume', 'Allocated Cost', 'Avg. Price', 'Actual Removal Vol %', 'Target Removal Vol %'])

    project_data_selected = project_data[project_data['project name'].isin(selected_project_names)].copy()

    if project_data_selected.empty:
        st.warning("Selected projects not found in the provided data.")
        return {}, pd.DataFrame(columns=['Year', f'Target {constraint_type}', 'Allocated Volume', 'Allocated Cost', 'Avg. Price', 'Actual Removal Vol %', 'Target Removal Vol %'])

    required_base_cols = ['project name', 'project type', 'priority']
    price_cols_needed, volume_cols_needed = [], []
    for year_alloc_val in selected_years: # Renamed year
        price_cols_needed.append(f"price {year_alloc_val}")
        volume_cols_needed.append(f"available volume {year_alloc_val}")

    missing_base = [col for col in required_base_cols if col not in project_data_selected.columns]
    if missing_base:
        raise ValueError(f"Input data is missing required base columns for allocation: {', '.join(missing_base)}")

    missing_years_data = []
    for year_alloc_val_check in selected_years: # Renamed year
        if f"price {year_alloc_val_check}" not in project_data_selected.columns:
            missing_years_data.append(f"price {year_alloc_val_check}")
        if f"available volume {year_alloc_val_check}" not in project_data_selected.columns:
            missing_years_data.append(f"available volume {year_alloc_val_check}")

    if missing_years_data:
        years_affected = sorted(list(set(int(col.split()[-1]) for col in missing_years_data if col.split()[-1].isdigit())))
        raise ValueError(f"Input data (for selected projects) is missing price/volume information for required year(s): {', '.join(map(str, years_affected))}.")

    numeric_cols_to_check = ['priority'] + price_cols_needed + volume_cols_needed
    for col in numeric_cols_to_check:
        if col in project_data_selected.columns:
            project_data_selected[col] = pd.to_numeric(project_data_selected[col], errors='coerce')
            if col == 'priority':
                project_data_selected[col] = project_data_selected[col].fillna(0) 
            elif col.startswith("available volume"):
                project_data_selected[col] = project_data_selected[col].fillna(0).apply(lambda x: int(x) if pd.notna(x) else 0).clip(lower=0)
            elif col.startswith("price"):
                project_data_selected[col] = project_data_selected[col].fillna(0.0).apply(lambda x: float(x) if pd.notna(x) else 0.0).clip(lower=0.0)

    all_project_types_in_selection = project_data_selected['project type'].unique()
    is_reduction_selected = 'reduction' in all_project_types_in_selection
    total_years_duration = end_year_portfolio - start_year_portfolio if end_year_portfolio >= start_year_portfolio else 0


    for year_loop in selected_years: # Renamed year
        yearly_target_val = annual_targets.get(year_loop, 0) # Renamed yearly_target
        price_col_loop = f"price {year_loop}" # Renamed price_col
        volume_col_loop = f"available volume {year_loop}" # Renamed volume_col

        year_total_allocated_vol = 0
        year_total_allocated_cost = 0.0
        summary_template = {
            'Year': year_loop, f'Target {constraint_type}': yearly_target_val, 'Allocated Volume': 0,
            'Allocated Cost': 0.0, 'Avg. Price': 0.0, 'Actual Removal Vol %': 0.0,
            'Target Removal Vol %': 0.0 
        }

        if yearly_target_val <= 0:
            yearly_summary_list.append(summary_template); portfolio_details[year_loop] = []; continue

        target_percentages = {}
        if is_reduction_selected:
            start_removal_percent = 0.10; end_removal_percent = removal_target_percent_end_year
            progress = 1.0 if total_years_duration <= 0 else max(0, min(1, (year_loop - start_year_portfolio) / total_years_duration))
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
            if total_alloc_share > 1e-9: # Normalize shares if both types are selected
                if tech_selected : target_percentages['technical removal'] = tech_alloc_share / total_alloc_share 
                else: target_percentages['technical removal'] = 0.0
                if nat_selected : target_percentages['natural removal'] = nat_alloc_share / total_alloc_share
                else: target_percentages['natural removal'] = 0.0
            elif tech_selected: # Only tech selected
                 target_percentages['technical removal'] = 1.0; target_percentages['natural removal'] = 0.0
            elif nat_selected: # Only nat selected
                 target_percentages['natural removal'] = 1.0; target_percentages['technical removal'] = 0.0
            else: # No removal types selected (should not happen if logic is correct and projects are filtered)
                target_percentages['technical removal'] = 0.0; target_percentages['natural removal'] = 0.0
            target_percentages['reduction'] = 0.0


        current_sum = sum(target_percentages.values())
        if abs(current_sum - 1.0) > 1e-6 and current_sum > 0: # Normalize if sum is not 1
            norm_factor = 1.0 / current_sum
            target_percentages = {ptype: share * norm_factor for ptype, share in target_percentages.items()}
        
        summary_template['Target Removal Vol %'] = (target_percentages.get('technical removal', 0) + target_percentages.get('natural removal', 0)) * 100

        projects_year_df = project_data_selected[(project_data_selected[price_col_loop] > 0) & (project_data_selected[volume_col_loop] >= min_allocation_chunk)].copy()
        projects_year_df['initial_allocated_volume'] = 0 
        projects_year_df['initial_allocated_cost'] = 0.0
        projects_year_df['final_priority'] = np.nan # Initialize

        if projects_year_df.empty:
            yearly_summary_list.append(summary_template); portfolio_details[year_loop] = []; continue

        for project_type_alloc in all_project_types_in_selection: # Renamed project_type
            target_share = target_percentages.get(project_type_alloc, 0)
            if target_share <= 0: continue
            
            target_resource_for_type = yearly_target_val * target_share # Renamed target_resource
            projects_of_this_type = projects_year_df[projects_year_df['project type'] == project_type_alloc].copy() # Renamed projects_of_type

            if projects_of_this_type.empty: continue

            total_priority_in_type = projects_of_this_type['priority'].sum()
            if total_priority_in_type <= 0: # Equal weighting if no priorities or sum is zero
                num_projects_in_type = len(projects_of_this_type)
                projects_of_this_type['norm_prio_base'] = (1.0 / num_projects_in_type) if num_projects_in_type > 0 else 0
            else: 
                projects_of_this_type['norm_prio_base'] = projects_of_this_type['priority'] / total_priority_in_type
            
            current_priorities_dict = projects_of_this_type.set_index('project name')['norm_prio_base'].to_dict() # Renamed current_priorities
            final_priorities_dict = current_priorities_dict.copy() # Renamed final_priorities

            if favorite_project and favorite_project in final_priorities_dict:
                fav_proj_base_prio = current_priorities_dict[favorite_project]
                boost_factor = priority_boost_percent / 100.0
                priority_increase = fav_proj_base_prio * boost_factor
                new_fav_proj_prio = fav_proj_base_prio + priority_increase
                
                other_projects_names = [p_name for p_name in current_priorities_dict if p_name != favorite_project] # Renamed other_projects
                sum_other_priorities_val = sum(current_priorities_dict[p_name_other] for p_name_other in other_projects_names) # Renamed sum_other_priorities, p_name

                temp_priorities_calc = {favorite_project: new_fav_proj_prio} # Renamed temp_priorities
                reduction_factor_calc = 0 # Renamed reduction_factor
                if sum_other_priorities_val > 1e-9: # Avoid division by zero
                    reduction_factor_calc = priority_increase / sum_other_priorities_val
                
                for name_other_proj in other_projects_names: # Renamed name
                    temp_priorities_calc[name_other_proj] = max(0, current_priorities_dict[name_other_proj] * (1 - reduction_factor_calc))
                
                total_final_prio_sum = sum(temp_priorities_calc.values()) # Renamed total_final_prio
                if total_final_prio_sum > 1e-9:
                    final_priorities_dict = {p_name_final: prio_final / total_final_prio_sum for p_name_final, prio_final in temp_priorities_calc.items()} # Renamed p, prio
                elif favorite_project in temp_priorities_calc : # Only favorite project has priority
                     final_priorities_dict = {favorite_project: 1.0}
                     for p_name_other_final in other_projects_names: final_priorities_dict[p_name_other_final] = 0.0


            project_weights_calc = {} # Renamed project_weights
            total_weight_calc = 0 # Renamed total_weight
            if constraint_type == 'Budget':
                for _, row_budget in projects_of_this_type.iterrows(): # Renamed row
                    name_budget = row_budget['project name'] # Renamed name
                    final_prio_budget = final_priorities_dict.get(name_budget, 0) # Renamed final_prio
                    price_budget = row_budget[price_col_loop] # Renamed price
                    weight_val = final_prio_budget * price_budget if price_budget > 0 else 0 # Renamed weight
                    project_weights_calc[name_budget] = weight_val
                    total_weight_calc += weight_val
            
            for idx, row_alloc in projects_of_this_type.iterrows(): # Renamed row
                name_alloc = row_alloc['project name'] # Renamed name
                final_prio_alloc = final_priorities_dict.get(name_alloc, 0) # Renamed final_prio
                available_vol_alloc = row_alloc[volume_col_loop] # Renamed available_vol
                price_alloc = row_alloc[price_col_loop] # Renamed price
                
                allocated_volume_proj = 0 # Renamed allocated_volume
                # allocated_cost_proj = 0.0 # Renamed allocated_cost - not needed here, calculated from volume

                projects_year_df.loc[projects_year_df['project name'] == name_alloc, 'final_priority'] = final_prio_alloc
                
                if final_prio_alloc <= 0 or price_alloc <= 0 or available_vol_alloc < min_allocation_chunk:
                    continue

                if constraint_type == 'Volume':
                    target_volume_proj_calc = target_resource_for_type * final_prio_alloc # Renamed target_volume_proj
                    allocated_volume_proj = min(target_volume_proj_calc, available_vol_alloc)
                elif constraint_type == 'Budget':
                    if total_weight_calc > 1e-9:
                        weight_normalized_calc = project_weights_calc.get(name_alloc, 0) / total_weight_calc # Renamed weight_normalized
                        target_budget_proj_calc = target_resource_for_type * weight_normalized_calc # Renamed target_budget_proj
                        target_volume_from_budget = target_budget_proj_calc / price_alloc if price_alloc > 0 else 0 # Renamed target_volume_proj
                        allocated_volume_proj = min(target_volume_from_budget, available_vol_alloc)
                    else: # No weight, typically means no projects with price or priority in this type
                        allocated_volume_proj = 0
                
                allocated_volume_proj = int(max(0, math.floor(allocated_volume_proj)))
                
                if allocated_volume_proj >= min_allocation_chunk:
                    allocated_cost_final = allocated_volume_proj * price_alloc # Renamed allocated_cost
                    projects_year_df.loc[projects_year_df['project name'] == name_alloc, 'initial_allocated_volume'] += allocated_volume_proj
                    projects_year_df.loc[projects_year_df['project name'] == name_alloc, 'initial_allocated_cost'] += allocated_cost_final
                    year_total_allocated_vol += allocated_volume_proj
                    year_total_allocated_cost += allocated_cost_final

        target_threshold_adj = yearly_target_val * min_target_fulfillment_percent # Renamed target_threshold
        current_metric_total_adj = year_total_allocated_cost if constraint_type == 'Budget' else year_total_allocated_vol # Renamed current_metric_total

        if current_metric_total_adj < target_threshold_adj and yearly_target_val > 0:
            needed_adj = target_threshold_adj - current_metric_total_adj # Renamed needed
            projects_year_df['remaining_volume'] = projects_year_df[volume_col_loop] - projects_year_df['initial_allocated_volume']
            
            adjustment_candidates_df = projects_year_df[ # Renamed adjustment_candidates
                (projects_year_df['remaining_volume'] >= min_allocation_chunk) & 
                (projects_year_df[price_col_loop] > 0)
            ].sort_values(by=['final_priority', 'priority'], ascending=[False, False]).copy() # Sort by final_priority (if exists), then original

            for idx_adj, row_adj in adjustment_candidates_df.iterrows(): # Renamed idx, row
                if needed_adj <= 1e-6: break # Using a small epsilon for float comparison
                
                name_adj = row_adj['project name'] # Renamed name
                price_adj = row_adj[price_col_loop] # Renamed price
                available_for_adj_vol = row_adj['remaining_volume'] # Renamed available_for_adj
                
                volume_to_add_iter = 0 # Renamed volume_to_add
                cost_to_add_iter = 0.0 # Renamed cost_to_add

                if constraint_type == 'Volume':
                    add_vol_calc = int(math.floor(min(available_for_adj_vol, needed_adj))) # Renamed add_vol
                else: # Budget constraint
                    max_affordable_vol_calc = int(math.floor(needed_adj / price_adj)) if price_adj > 0 else 0 # Renamed max_affordable_vol
                    add_vol_calc = min(available_for_adj_vol, max_affordable_vol_calc)
                
                if add_vol_calc >= min_allocation_chunk:
                    cost_increase_val = add_vol_calc * price_adj # Renamed cost_increase
                    # Check if adding this chunk is worthwhile or doesn't grossly overshoot budget component
                    if constraint_type == 'Volume' or \
                       (constraint_type == 'Budget' and cost_increase_val <= needed_adj * 1.10) or \
                       (cost_increase_val < price_adj * min_allocation_chunk * 1.5): # Allow small overshoot for min chunk
                        
                        volume_to_add_iter = add_vol_calc
                        cost_to_add_iter = cost_increase_val
                        
                        if constraint_type == 'Budget':
                            needed_adj -= cost_to_add_iter
                        else: # Volume constraint
                            needed_adj -= volume_to_add_iter
                
                if volume_to_add_iter > 0:
                    project_idx = projects_year_df.index[projects_year_df['project name'] == name_adj].tolist()[0] # Get original index
                    projects_year_df.loc[project_idx, 'initial_allocated_volume'] += volume_to_add_iter
                    projects_year_df.loc[project_idx, 'initial_allocated_cost'] += cost_to_add_iter
                    projects_year_df.loc[project_idx, 'remaining_volume'] -= volume_to_add_iter # Keep track
                    year_total_allocated_vol += volume_to_add_iter
                    year_total_allocated_cost += cost_to_add_iter

        final_allocations_list_year = [] # Renamed final_allocations_list
        final_year_allocations_df_calc = projects_year_df[projects_year_df['initial_allocated_volume'] >= min_allocation_chunk].copy() # Renamed final_year_allocations_df

        for _, row_final_alloc in final_year_allocations_df_calc.iterrows(): # Renamed idx, row
            current_price_final = row_final_alloc.get(price_col_loop, None) # Renamed current_price
            final_allocations_list_year.append({
                'project name': row_final_alloc['project name'], 
                'type': row_final_alloc['project type'], 
                'allocated_volume': row_final_alloc['initial_allocated_volume'], 
                'allocated_cost': row_final_alloc['initial_allocated_cost'], 
                'price_used': current_price_final, 
                'priority_applied': row_final_alloc['final_priority']
            })
        portfolio_details[year_loop] = final_allocations_list_year
        
        summary_template['Allocated Volume'] = year_total_allocated_vol
        summary_template['Allocated Cost'] = year_total_allocated_cost
        summary_template['Avg. Price'] = (year_total_allocated_cost / year_total_allocated_vol) if year_total_allocated_vol > 0 else 0.0
        
        removal_volume_calc = sum(p_item['allocated_volume'] for p_item in final_allocations_list_year if p_item['type'] in ['technical removal', 'natural removal']) # Renamed removal_volume, p
        summary_template['Actual Removal Vol %'] = (removal_volume_calc / year_total_allocated_vol * 100) if year_total_allocated_vol > 0 else 0.0
        yearly_summary_list.append(summary_template)

    yearly_summary_df_final = pd.DataFrame(yearly_summary_list) # Renamed yearly_summary_df

    if constraint_type == 'Budget':
        check_df_budget = yearly_summary_df_final.copy() # Renamed check_df
        check_df_budget['Target Budget'] = check_df_budget['Year'].map(annual_targets).fillna(0)
        is_overbudget_check = check_df_budget['Allocated Cost'] > check_df_budget['Target Budget'] * 1.001 # Renamed is_overbudget
        overbudget_years_df_check = check_df_budget[is_overbudget_check] # Renamed overbudget_years_df
        if not overbudget_years_df_check.empty:
            st.warning(f"Budget target may have been slightly exceeded in year(s): {overbudget_years_df_check['Year'].tolist()} due to allocation adjustments or minimum chunk requirements.")

    return portfolio_details, yearly_summary_df_final


# ==================================
# Streamlit App Layout & Logic
# ==================================
with st.sidebar:
    st.markdown("## 1. Load Data")
    df_upload = st.file_uploader("Upload Project Data CSV", type="csv", key="uploader_sidebar", help="CSV required columns: `project name`, `project type` ('reduction', 'technical removal', 'natural removal'), `priority`. Also needs `price_YYYY` and `available_volume_YYYY` columns for relevant years. Optional: `description`, `project_link`.")
    
    # Initialize session state variables if they don't exist
    default_values = {
        'working_data_full': None, 'selected_years': [], 'selected_projects': [], 
        'project_names': [], 'favorite_projects_selection': [], 
        'actual_start_year': None, 'actual_end_year': None, 
        'available_years_in_data': [], 'constraint_type': 'Volume', 
        'removal_target_end_year': 0.8, 'transition_speed': 5, 
        'category_split': {'technical removal': 0.5, 'natural removal': 0.5}, 
        'annual_targets': {}, 'master_target': None, 
        'data_loaded_successfully': False, 'years_slider_sidebar': 5, 
        'min_fulfillment_perc': 95, 'removal_preference_slider': 5, 'min_alloc_chunk': 1
    }
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    if df_upload:
        try:
            # Call the refactored data loading function
            data, error_msg, available_years_in_data, project_names_list, invalid_types_found = load_and_prepare_data_simplified(df_upload)

            if invalid_types_found:
                st.sidebar.warning(f"Ignored rows with invalid project types: {', '.join(invalid_types_found)}. Valid types are 'reduction', 'technical removal', 'natural removal'.")
            
            if error_msg:
                st.sidebar.error(error_msg) # Display the detailed error message
                # Reset states on error
                st.session_state.data_loaded_successfully = False
                st.session_state.working_data_full = None
                st.session_state.project_names = []
                st.session_state.available_years_in_data = []
                st.session_state.selected_projects = []
                st.session_state.annual_targets = {}
            else:
                # Successfully loaded and prepared data
                st.session_state.project_names = project_names_list
                st.session_state.available_years_in_data = available_years_in_data
                st.session_state.working_data_full = data
                st.session_state.data_loaded_successfully = True
                st.sidebar.success("Data loaded successfully!")

                # Update selected_projects: if previous selection is now invalid or empty, default to all available
                current_selection = st.session_state.get('selected_projects', [])
                valid_current_selection = [p for p in current_selection if p in project_names_list]
                if not valid_current_selection and project_names_list:
                    st.session_state.selected_projects = project_names_list 
                else:
                    st.session_state.selected_projects = valid_current_selection
                
                # Reset annual targets for the new dataset
                st.session_state.annual_targets = {} 
                st.session_state.master_target = None # Reset master target as well
        
        except Exception as e:
            st.sidebar.error(f"An unexpected critical error occurred during file processing: {e}")
            st.sidebar.error(f"Traceback: {traceback.format_exc()}")
            # Ensure all relevant states are reset
            st.session_state.data_loaded_successfully = False
            st.session_state.working_data_full = None
            st.session_state.project_names = []
            st.session_state.available_years_in_data = []
            st.session_state.selected_projects = []
            st.session_state.annual_targets = {}


    # --- Settings (Displayed only if data loaded successfully) ---
    if st.session_state.get('data_loaded_successfully', False) and st.session_state.working_data_full is not None:
        data_for_ui = st.session_state.working_data_full
        available_years_from_data = st.session_state.available_years_in_data # Renamed
        project_names_from_data = st.session_state.project_names # Renamed

        if not available_years_from_data:
            st.sidebar.warning("No usable year data (e.g. price_YYYY, available_volume_YYYY columns) found in the uploaded file.")
            st.session_state.selected_years = []
            st.session_state.actual_start_year = None
            st.session_state.actual_end_year = None
        else:
            st.markdown("## 2. Portfolio Settings")
            min_year_data_ui = min(available_years_from_data) # Renamed
            max_year_data_ui = max(available_years_from_data) # Renamed
            max_years_slider_limit = min(20, max(1, max_year_data_ui - min_year_data_ui + 1)) # Renamed

            current_slider_val_ui = st.session_state.get('years_slider_sidebar', 5) # Renamed
            if current_slider_val_ui > max_years_slider_limit : current_slider_val_ui = max_years_slider_limit
            if current_slider_val_ui < 1: current_slider_val_ui = 1

            years_to_plan_slider = st.slider( # Renamed
                f"Years to Plan (Starting {min_year_data_ui})",
                1, max_years_slider_limit,
                value=current_slider_val_ui, 
                key='years_slider_sidebar_widget_unique' # Ensured key is unique
            )
            st.session_state.years_slider_sidebar = years_to_plan_slider

            start_year_selected_ui = min_year_data_ui # Renamed
            end_year_selected_ui = start_year_selected_ui + years_to_plan_slider - 1 # Renamed

            actual_years_present_in_data_ui = [] # Renamed
            for year_iter_ui in range(start_year_selected_ui, end_year_selected_ui + 1): # Renamed
                price_col_ui = f"price {year_iter_ui}" # Renamed
                vol_col_ui = f"available volume {year_iter_ui}" # Renamed
                if price_col_ui in data_for_ui.columns and vol_col_ui in data_for_ui.columns:
                    actual_years_present_in_data_ui.append(year_iter_ui)

            st.session_state.selected_years = actual_years_present_in_data_ui

            if not st.session_state.selected_years:
                st.sidebar.error(f"No data available for the selected period ({start_year_selected_ui}-{end_year_selected_ui}). Adjust 'Years to Plan' or check CSV.")
                st.session_state.actual_start_year = None
                st.session_state.actual_end_year = None
            else:
                st.session_state.actual_start_year = min(st.session_state.selected_years)
                st.session_state.actual_end_year = max(st.session_state.selected_years)
                st.sidebar.markdown(f"Planning Horizon: **{st.session_state.actual_start_year} - {st.session_state.actual_end_year}**")

                # Retrieve constraint_type from session_state, default to 'Volume'
                current_constraint_type = st.session_state.get('constraint_type', 'Volume')
                constraint_type_index = ['Volume', 'Budget'].index(current_constraint_type) if current_constraint_type in ['Volume', 'Budget'] else 0
                
                st.session_state.constraint_type = st.radio(
                    "Constraint Type:", ('Volume', 'Budget'),
                    index=constraint_type_index,
                    key='constraint_type_sidebar_widget_unique', 
                    horizontal=True,
                    help="Choose annual target units: tons (Volume) or currency (Budget)."
                )
                # constraint_type_for_logic = st.session_state.constraint_type # Use this for immediate logic

                st.markdown("### Annual Target Settings")
                # Use a different key for master_target to avoid conflict if type changes
                master_target_key_suffix = "_vol" if st.session_state.constraint_type == 'Volume' else "_bud"
                master_target_session_key = f'master_target{master_target_key_suffix}'
                
                master_target_value_ui = st.session_state.get(master_target_session_key) # Renamed

                if st.session_state.constraint_type == 'Volume':
                    default_val_vol_ui = 1000 # Renamed
                    if master_target_value_ui is not None and isinstance(master_target_value_ui, (int, float)):
                        try: default_val_vol_ui = int(float(master_target_value_ui))
                        except (ValueError, TypeError): pass # Keep default if conversion fails
                    
                    default_target_input = st.number_input( # Renamed
                        "Default Annual Target Volume (t):", min_value=0, step=100,
                        value=default_val_vol_ui, key='master_target_input_vol', # Unique key for volume
                        help="Default target volume/year. Override specific years below."
                    )
                    st.session_state[master_target_session_key] = default_target_input # Store with type-specific key
                    st.session_state.master_target = default_target_input # Generic key for use in allocation

                else: # Budget constraint
                    default_val_bud_ui = 100000.0 # Renamed
                    if master_target_value_ui is not None and isinstance(master_target_value_ui, (int, float)):
                        try: default_val_bud_ui = float(master_target_value_ui)
                        except (ValueError, TypeError): pass
                    
                    default_target_input = st.number_input(
                        "Default Annual Target Budget (€):", min_value=0.0, step=1000.0,
                        value=default_val_bud_ui, format="%.2f", key='master_target_input_bud', # Unique key for budget
                        help="Default target budget/year. Override specific years below."
                    )
                    st.session_state[master_target_session_key] = default_target_input # Store with type-specific key
                    st.session_state.master_target = default_target_input # Generic key

                with st.expander("Customize Annual Targets (+/-)", expanded=False):
                    current_annual_targets_ui = st.session_state.get('annual_targets', {}) # Renamed
                    updated_targets_from_inputs_ui = {} # Renamed

                    if not st.session_state.selected_years:
                        st.caption("Select 'Years to Plan' first.")
                    else:
                        for year_custom_target in st.session_state.selected_years: # Renamed
                            # Use master_target (generic one) as default for individual years
                            year_target_value_default = current_annual_targets_ui.get(year_custom_target, st.session_state.master_target)
                            
                            input_key_custom = f"target_{year_custom_target}_{st.session_state.constraint_type}" # Renamed
                            label_custom = f"Target {year_custom_target} [t]" if st.session_state.constraint_type == 'Volume' else f"Target {year_custom_target} [€]" # Renamed

                            if st.session_state.constraint_type == 'Volume':
                                try: input_val_vol = int(year_target_value_default) # Renamed
                                except (ValueError, TypeError): input_val_vol = int(st.session_state.master_target if isinstance(st.session_state.master_target, (int,float)) else 1000)
                                updated_targets_from_inputs_ui[year_custom_target] = st.number_input(
                                    label_custom, min_value=0, step=100, value=input_val_vol, key=input_key_custom
                                )
                            else: # Budget
                                try: input_val_bud = float(year_target_value_default) # Renamed
                                except (ValueError, TypeError): input_val_bud = float(st.session_state.master_target if isinstance(st.session_state.master_target, (int,float)) else 100000.0)
                                updated_targets_from_inputs_ui[year_custom_target] = st.number_input(
                                    label_custom, min_value=0.0, step=1000.0, value=input_val_bud, format="%.2f", key=input_key_custom
                                )
                    st.session_state.annual_targets = updated_targets_from_inputs_ui

                st.sidebar.markdown("### Allocation Goal & Preferences")
                min_fulfill_slider = st.sidebar.slider(f"Min. Target Fulfillment (%)", 50, 100, st.session_state.get('min_fulfillment_perc', 95), help=f"Attempt >= this % of target {st.session_state.constraint_type} via adjustment.", key='min_fulfill_perc_sidebar_unique') # Renamed
                st.session_state.min_fulfillment_perc = min_fulfill_slider
                
                min_chunk_input = st.sidebar.number_input("Min. Allocation Unit (t)", min_value=1, step=1, value=st.session_state.get('min_alloc_chunk', 1), help="Smallest amount (tons) to allocate per project/year.", key='min_alloc_chunk_sidebar_unique') # Renamed
                st.session_state.min_alloc_chunk = int(min_chunk_input)

                st.sidebar.markdown("### Removal Volume Transition (If 'Reduction' Projects Used)")
                reduction_present_ui = 'reduction' in data_for_ui['project type'].unique() if data_for_ui is not None and 'project type' in data_for_ui.columns else False # Renamed

                if reduction_present_ui: st.sidebar.info("Transition settings apply if 'Reduction' projects are selected and used.")
                else: st.sidebar.info("Transition inactive: No 'Reduction' projects found or selected.")
                
                actual_end_year_for_help = st.session_state.actual_end_year if st.session_state.actual_end_year else 'N/A'
                removal_help_text = f"Target % vol from Removals in final year ({actual_end_year_for_help}). Guides mix if 'Reduction' selected." # Renamed
                
                current_removal_target_slider_val = int(st.session_state.get('removal_target_end_year', 0.8)*100)
                rem_target_slider_widget = st.sidebar.slider(f"Target Removal Vol % ({actual_end_year_for_help})", 0, 100, current_removal_target_slider_val, help=removal_help_text, key='removal_perc_slider_sidebar_unique', disabled=not reduction_present_ui or not st.session_state.actual_end_year) # Renamed
                st.session_state.removal_target_end_year = rem_target_slider_widget / 100.0
                
                st.session_state.transition_speed = st.sidebar.slider("Transition Speed", 1, 10, st.session_state.get('transition_speed', 5), help="Ramp-up speed (1=Slow, 10=Fast) if Reductions selected.", key='transition_speed_slider_sidebar_unique', disabled=not reduction_present_ui)

                st.sidebar.markdown("### Removal Category Preference")
                removal_types_present_ui = any(pt in data_for_ui['project type'].unique() for pt in ['technical removal', 'natural removal']) if data_for_ui is not None and 'project type' in data_for_ui.columns else False # Renamed

                rem_pref_slider_widget = st.sidebar.slider("Technical vs Natural Preference", 1, 10, st.session_state.get('removal_preference_slider', 5), format="%d", key='removal_pref_slider_sidebar_unique', help="1 leans Natural, 5 balanced, 10 leans Technical.", disabled=not removal_types_present_ui) # Renamed
                st.session_state['removal_preference_slider'] = rem_pref_slider_widget
                tech_pref_ratio_calc = (rem_pref_slider_widget - 1) / 9.0 # Renamed
                st.session_state.category_split = {'technical removal': tech_pref_ratio_calc, 'natural removal': 1.0 - tech_pref_ratio_calc}

                st.sidebar.markdown("## 3. Select Projects")
                if not project_names_from_data:
                    st.sidebar.warning("No projects available from loaded data.")
                else:
                    current_selected_projects_val = st.session_state.get('selected_projects', []) # Renamed
                    valid_defaults_projects = [p_name for p_name in current_selected_projects_val if p_name in project_names_from_data] # Renamed
                    if not valid_defaults_projects and project_names_from_data: 
                        valid_defaults_projects = project_names_from_data

                    st.session_state.selected_projects = st.sidebar.multiselect(
                        "Select projects to include:", options=project_names_from_data,
                        default=valid_defaults_projects, key='project_selector_sidebar_unique'
                    )

                    if 'priority' in data_for_ui.columns: 
                        boost_options_list = [p_name for p_name in project_names_from_data if p_name in st.session_state.selected_projects] # Renamed
                        if boost_options_list:
                            current_favorite_val = st.session_state.get('favorite_projects_selection', []) # Renamed
                            valid_default_favorite_proj = [fav for fav in current_favorite_val if fav in boost_options_list][:1] # Renamed
                            st.session_state.favorite_projects_selection = st.sidebar.multiselect(
                                "Favorite Project (Priority Boost):", options=boost_options_list, 
                                default=valid_default_favorite_proj, key='favorite_selector_sidebar_unique',
                                max_selections=1, help="Boost priority for one project."
                            )
                        else:
                            st.sidebar.info("Select projects from the list above to enable priority boost.")
                            st.session_state.favorite_projects_selection = [] 
                    else:
                        st.sidebar.info("Priority boost disabled: No 'priority' column in data.")
                        st.session_state.favorite_projects_selection = [] 

# ==================================
# Main Page Content
# ==================================
st.markdown(f"<h1 style='color: #8ca734;'>Carbon Portfolio Builder</h1>", unsafe_allow_html=True)
st.markdown("---")

if not st.session_state.get('data_loaded_successfully', False):
    st.info("👋 Welcome! Please upload your project data CSV using the sidebar menu to get started.")
    st.caption("Make sure your CSV has columns like: `project_name`, `project_type`, `priority`, and then year-specific columns like `price_2024`, `available_volume_2024`, etc.")

elif st.session_state.get('data_loaded_successfully', False) and st.session_state.working_data_full is not None:
    st.markdown("## Project Offerings") 
    st.caption("Review available projects from your uploaded data. Use the sidebar to configure portfolio settings and select projects.") 

    project_df_display_main = st.session_state.working_data_full.copy() # Renamed
    available_years_main_display = st.session_state.available_years_in_data # Renamed
    
    price_cols_for_avg = [f"price {year_avg}" for year_avg in available_years_main_display if f"price {year_avg}" in project_df_display_main.columns] # Renamed

    if price_cols_for_avg: 
        project_df_display_main['Average Price'] = project_df_display_main[price_cols_for_avg].mean(axis=1, skipna=True).fillna(0.0)
    else:
        project_df_display_main['Average Price'] = 0.0 

    with st.expander("View Project Details and Average Prices"):
        if not project_df_display_main.empty:
            display_cols_main = ['project name', 'project type', 'Average Price'] # Renamed
            column_config_main = { # Renamed
                    "Average Price": st.column_config.NumberColumn("Avg Price (€/t)", help="Average price across all available years in input data.", format="€%.2f")
            }
            if 'Description' in project_df_display_main.columns: # Check for 'Description' (after rename)
                display_cols_main.insert(2, 'Description') # Add description before average price
                column_config_main["Description"] = st.column_config.TextColumn("Description", width="medium")

            if 'Project Link' in project_df_display_main.columns: 
                display_cols_main.append('Project Link')
                column_config_main["Project Link"] = st.column_config.LinkColumn("Project Link", display_text="Visit ->", help="Link to project page (if available).")

            st.dataframe(project_df_display_main[display_cols_main], column_config=column_config_main, hide_index=True, use_container_width=True)
        else:
            st.write("Project data is loaded but appears to be empty or no projects with valid types were found.")
    st.markdown("---") 


    if not st.session_state.get('selected_projects'):
        st.warning("⚠️ Please select projects from the sidebar (Section 3) to build a portfolio.")
    elif not st.session_state.get('selected_years'):
        st.warning("⚠️ No valid years for planning. Adjust 'Years to Plan' in sidebar (Section 2) or check CSV data.")
    else:
        # Check all required keys for allocation are present and reasonably filled
        required_keys_for_alloc = [ # Renamed
            'working_data_full', 'selected_projects', 'selected_years',
            'actual_start_year', 'actual_end_year', 'constraint_type', 
            'annual_targets', 'removal_target_end_year', 'transition_speed', 
            'category_split', 'favorite_projects_selection', 
            'min_fulfillment_perc', 'min_alloc_chunk', 'master_target'
        ]
        # Ensure all keys exist and critical ones are not None
        keys_are_present = all(k in st.session_state for k in required_keys_for_alloc)
        if keys_are_present: # Further checks for None where applicable
            if any(st.session_state.get(k) is None for k in ['working_data_full', 'selected_projects', 'selected_years', 'actual_start_year', 'actual_end_year', 'constraint_type', 'master_target']):
                keys_are_present = False
        

        if keys_are_present:
            try:
                fav_proj_alloc = st.session_state.favorite_projects_selection[0] if st.session_state.favorite_projects_selection else None # Renamed
                constraint_alloc = st.session_state.constraint_type # Renamed
                min_chunk_alloc = st.session_state.min_alloc_chunk # Renamed
                
                annual_targets_for_alloc = st.session_state.annual_targets.copy() # Renamed
                # If annual_targets is empty but master_target and selected_years exist, populate it
                if not annual_targets_for_alloc and st.session_state.selected_years and st.session_state.master_target is not None:
                    annual_targets_for_alloc = {year_val_at: st.session_state.master_target for year_val_at in st.session_state.selected_years} # Renamed

                if not annual_targets_for_alloc and st.session_state.selected_years:
                     st.warning("Annual targets are not set. Portfolio allocation might be zero. Please set a default annual target or customize per year in the sidebar.")


                if constraint_alloc == 'Budget': st.info(f"**Budget Mode:** Projects receive budget based on weighted priority and price. A project might get zero volume if its minimum allocation ({min_chunk_alloc}t) costs more than its allocated budget share. The adjustment step may allocate additional volume later.")
                st.success(f"**Allocation Goal:** Attempting to achieve at least **{st.session_state.min_fulfillment_perc}%** of the annual target for {constraint_alloc}.")

                with st.spinner("Calculating portfolio allocation... This may take a moment."):
                    results_alloc, summary_alloc = allocate_portfolio( # Renamed
                        project_data=st.session_state.working_data_full,
                        selected_project_names=st.session_state.selected_projects,
                        selected_years=st.session_state.selected_years,
                        start_year_portfolio=st.session_state.actual_start_year,
                        end_year_portfolio=st.session_state.actual_end_year,
                        constraint_type=constraint_alloc,
                        annual_targets=annual_targets_for_alloc, 
                        removal_target_percent_end_year=st.session_state.removal_target_end_year,
                        transition_speed=st.session_state.transition_speed,
                        category_split=st.session_state.category_split,
                        favorite_project=fav_proj_alloc,
                        priority_boost_percent=10, 
                        min_target_fulfillment_percent=st.session_state.min_fulfillment_perc / 100.0,
                        min_allocation_chunk=min_chunk_alloc
                    )

                details_list_main = [] # Renamed
                if results_alloc:
                    for year_res, projects_res_list in results_alloc.items(): # Renamed
                        if projects_res_list: 
                            for proj_item_res in projects_res_list: # Renamed
                                if isinstance(proj_item_res, dict) and (proj_item_res.get('allocated_volume', 0) >= min_chunk_alloc or proj_item_res.get('allocated_cost', 0) > 1e-6):
                                    details_list_main.append({
                                        'year': year_res,
                                        'project name': proj_item_res.get('project name'),
                                        'type': proj_item_res.get('type'),
                                        'volume': proj_item_res.get('allocated_volume', 0),
                                        'price': proj_item_res.get('price_used', None), 
                                        'cost': proj_item_res.get('allocated_cost', 0.0)
                                    })
                details_df_main = pd.DataFrame(details_list_main) # Renamed
                pivot_display_main = pd.DataFrame() # Renamed

                st.markdown("## Portfolio Summary")
                col_l_main, col_r_main = st.columns([2, 1.2], gap="large") # Renamed

                with col_l_main:
                    st.markdown("#### Key Metrics (Overall)")
                    if not summary_alloc.empty:
                        total_cost_all_years_sum = summary_alloc['Allocated Cost'].sum() # Renamed
                        total_volume_all_years_sum = summary_alloc['Allocated Volume'].sum() # Renamed
                        overall_avg_price_sum = total_cost_all_years_sum / total_volume_all_years_sum if total_volume_all_years_sum > 0 else 0.0 # Renamed
                        st.markdown(f"""<div class="metric-box"><b>Total Portfolio Cost</b> € {total_cost_all_years_sum:,.2f}</div>""", unsafe_allow_html=True)
                        st.markdown(f"""<div class="metric-box"><b>Total Portfolio Volume</b> {total_volume_all_years_sum:,.0f} t</div>""", unsafe_allow_html=True)
                        st.markdown(f"""<div class="metric-box"><b>Overall Average Price</b> € {overall_avg_price_sum:,.2f} /t</div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="metric-box"><b>Total Portfolio Cost</b> - </div>""", unsafe_allow_html=True)
                        st.markdown(f"""<div class="metric-box"><b>Total Portfolio Volume</b> - </div>""", unsafe_allow_html=True)
                        st.markdown(f"""<div class="metric-box"><b>Overall Average Price</b> - </div>""", unsafe_allow_html=True)
                        st.warning("Overall portfolio metrics could not be calculated (summary table is empty).")

                with col_r_main:
                    st.markdown("#### Volume by Project Type")
                    if not details_df_main.empty:
                        pie_data_main = details_df_main.groupby('type')['volume'].sum().reset_index() # Renamed
                        pie_data_main = pie_data_main[pie_data_main['volume'] > 1e-6] 
                        if not pie_data_main.empty:
                            st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
                            fig_pie_main = px.pie(pie_data_main, values='volume', names='type', color='type', color_discrete_map=type_color_map) # Renamed
                            fig_pie_main.update_layout(
                                showlegend=True, legend_title_text='Project Type',
                                legend_orientation="h", legend_yanchor="bottom", legend_y=-0.2,
                                legend_xanchor="center", legend_x=0.5,
                                margin=dict(t=5, b=50, l=0, r=0), height=350
                            )
                            fig_pie_main.update_traces(textposition='inside', textinfo='percent+label', insidetextorientation='radial', sort=False, hole=.3, marker=dict(line=dict(color='#FFFFFF', width=1)))
                            st.plotly_chart(fig_pie_main, use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        elif not details_df_main.empty and pie_data_main.empty : 
                            st.caption("No significant volume allocated to any project type to display in pie chart.")
                    else:
                        st.caption("No detailed allocation data available to generate the volume by project type pie chart.")

                st.markdown("---") 

                if details_df_main.empty and summary_alloc.empty:
                    st.warning("No allocation data was generated. Cannot display further plots or tables.")
                # elif details_df_main.empty : # This condition is now handled by the above if summary_alloc is also empty.
                #     st.warning("No detailed project allocations available for the composition plot or detailed table.")
                else: # At least one of details_df_main or summary_alloc is not empty
                    st.markdown("### Portfolio Composition & Price Over Time")
                    
                    summary_plot_data_comp = pd.DataFrame() # Renamed
                    if not details_df_main.empty:
                        summary_plot_data_comp = details_df_main.groupby(['year', 'type']).agg(
                            volume=('volume', 'sum'), cost=('cost', 'sum')
                        ).reset_index()

                    price_summary_data_comp = pd.DataFrame() # Renamed
                    if not summary_alloc.empty:
                        price_summary_data_comp = summary_alloc[['Year', 'Avg. Price']].rename(columns={'Year':'year', 'Avg. Price':'avg_price'})

                    fig_composition_main = make_subplots(specs=[[{"secondary_y": True}]]) # Renamed
                    y_metric_comp = 'volume' if constraint_alloc == 'Volume' else 'cost' # Renamed
                    y_label_comp = 'Allocated Volume (t)' if constraint_alloc == 'Volume' else 'Allocated Cost (€)' # Renamed
                    y_format_comp = '{:,.0f}' if constraint_alloc == 'Volume' else '€{:,.2f}' # Renamed
                    y_hover_label_comp = 'Volume' if constraint_alloc == 'Volume' else 'Cost' # Renamed
                    
                    type_order_plot = ['reduction', 'natural removal', 'technical removal'] # Renamed
                    
                    if not summary_plot_data_comp.empty:
                        types_in_results_plot = summary_plot_data_comp['type'].unique() # Renamed
                        for t_name_plot in type_order_plot: # Renamed
                            if t_name_plot in types_in_results_plot:
                                df_type_plot = summary_plot_data_comp[summary_plot_data_comp['type'] == t_name_plot] # Renamed
                                if not df_type_plot.empty and y_metric_comp in df_type_plot.columns and df_type_plot[y_metric_comp].sum() > 1e-6:
                                    fig_composition_main.add_trace(
                                        go.Bar(
                                            x=df_type_plot['year'], y=df_type_plot[y_metric_comp], name=t_name_plot.replace('_', ' ').capitalize(),
                                            marker_color=type_color_map.get(t_name_plot, default_color),
                                            hovertemplate=f'Year: %{{x}}<br>Type: {t_name_plot.replace("_", " ").capitalize()}<br>{y_hover_label_comp}: %{{y:{y_format_comp}}}<extra></extra>'
                                        ), secondary_y=False
                                    )
                    else: # If details_df_main was empty, summary_plot_data_comp will be too.
                        st.caption("No detailed project data for composition bars.")


                    if not price_summary_data_comp.empty and 'avg_price' in price_summary_data_comp.columns:
                        fig_composition_main.add_trace(
                            go.Scatter(
                                x=price_summary_data_comp['year'], y=price_summary_data_comp['avg_price'], name='Avg Price (€/t)',
                                mode='lines+markers', marker=dict(symbol='circle', size=8), line=dict(color='#1B5E20', width=3),
                                hovertemplate='Year: %{x}<br>Avg Price: €%{y:,.2f}/t<extra></extra>'
                            ), secondary_y=True
                        )
                    if not summary_alloc.empty and 'Actual Removal Vol %' in summary_alloc.columns:
                        fig_composition_main.add_trace(
                            go.Scatter(
                                x=summary_alloc['Year'], y=summary_alloc['Actual Removal Vol %'], name='Actual Removal Vol %',
                                mode='lines+markers', line=dict(color='darkorange', dash='dash'), marker=dict(symbol='star', size=8),
                                hovertemplate='Year: %{x}<br>Actual Removal: %{y:.1f}%<extra></extra>'
                            ), secondary_y=True
                        )

                    y2_max_limit_plot = 105 # Renamed
                    if not price_summary_data_comp.empty and 'avg_price' in price_summary_data_comp.columns and not price_summary_data_comp['avg_price'].empty:
                        max_avg_price = price_summary_data_comp['avg_price'].max(skipna=True)
                        if pd.notna(max_avg_price): y2_max_limit_plot = max(y2_max_limit_plot, max_avg_price * 1.1)
                    if not summary_alloc.empty and 'Actual Removal Vol %' in summary_alloc.columns and not summary_alloc['Actual Removal Vol %'].empty:
                        max_actual_removal = summary_alloc['Actual Removal Vol %'].max(skipna=True)
                        if pd.notna(max_actual_removal): y2_max_limit_plot = max(y2_max_limit_plot, max_actual_removal * 1.1)

                    fig_composition_main.update_layout(
                        xaxis_title='Year', yaxis_title=y_label_comp, yaxis2_title='Avg Price (€/t) / Actual Removal %',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        barmode='stack', template="plotly_white", margin=dict(t=20, l=0, r=0, b=0),
                        yaxis=dict(rangemode='tozero'),
                        yaxis2=dict(rangemode='tozero', range=[0, y2_max_limit_plot if pd.notna(y2_max_limit_plot) and y2_max_limit_plot > 0 else 105]), 
                        hovermode="x unified"
                    )
                    if st.session_state.selected_years:
                        tick_interval = 1 if len(st.session_state.selected_years) < 20 else math.ceil(len(st.session_state.selected_years)/10)
                        fig_composition_main.update_xaxes(tickmode='array', tickvals=st.session_state.selected_years, dtick=tick_interval)
                    st.plotly_chart(fig_composition_main, use_container_width=True)

                st.markdown("### Detailed Allocation by Project and Year")
                if not details_df_main.empty or not summary_alloc.empty: 
                    try:
                        pivot_final_table = pd.DataFrame() # Renamed
                        
                        years_for_pivot_table = [] # Renamed
                        if not details_df_main.empty:
                            years_for_pivot_table = sorted(details_df_main['year'].unique())
                        elif not summary_alloc.empty:
                             years_for_pivot_table = sorted(summary_alloc['Year'].unique())
                        
                        if not years_for_pivot_table: # Fallback if both are empty but somehow this block is reached
                            years_for_pivot_table = st.session_state.selected_years


                        if not details_df_main.empty:
                            pivot_intermediate_table = pd.pivot_table(details_df_main, # Renamed
                                                                values=['volume', 'cost', 'price'],
                                                                index=['project name', 'type'], columns='year',
                                                                aggfunc={'volume': 'sum', 'cost': 'sum', 'price': 'first'}) 

                            if not pivot_intermediate_table.empty:
                                pivot_final_table = pivot_intermediate_table.swaplevel(0, 1, axis=1) 
                                metric_order_for_table = ['volume', 'cost', 'price'] # Renamed

                                final_multi_index_cols_for_table = pd.MultiIndex.from_product( # Renamed
                                    [years_for_pivot_table, metric_order_for_table], names=['year', 'metric']
                                )
                                pivot_final_table = pivot_final_table.reindex(columns=final_multi_index_cols_for_table).fillna(0) 
                                pivot_final_table = pivot_final_table.sort_index(axis=1, level=[0, 1]) 
                                pivot_final_table.index.names = ['Project Name', 'Type']
                        
                        total_data_for_table_row = {} # Renamed
                        if not summary_alloc.empty:
                            summary_indexed_for_table = summary_alloc.set_index('Year') # Renamed
                            for year_val_for_total_row in years_for_pivot_table: # Renamed
                                if year_val_for_total_row in summary_indexed_for_table.index:
                                    vol_total = summary_indexed_for_table.loc[year_val_for_total_row, 'Allocated Volume'] # Renamed
                                    cost_total = summary_indexed_for_table.loc[year_val_for_total_row, 'Allocated Cost'] # Renamed
                                    avg_price_total = summary_indexed_for_table.loc[year_val_for_total_row, 'Avg. Price'] # Renamed
                                else: 
                                    vol_total, cost_total, avg_price_total = 0, 0.0, 0.0
                                total_data_for_table_row[(year_val_for_total_row, 'volume')] = vol_total
                                total_data_for_table_row[(year_val_for_total_row, 'cost')] = cost_total
                                total_data_for_table_row[(year_val_for_total_row, 'price')] = avg_price_total 

                        total_row_index_for_df = pd.MultiIndex.from_tuples([('Total Portfolio', 'All Types')], names=['Project Name', 'Type']) # Renamed
                        total_row_df_for_concat = pd.DataFrame(total_data_for_table_row, index=total_row_index_for_df) # Renamed
                        
                        # Align columns of total_row_df_for_concat before concat or display
                        if years_for_pivot_table and not total_row_df_for_concat.empty:
                            metric_order_final_table = ['volume', 'cost', 'price']
                            expected_cols_total_row = pd.MultiIndex.from_product(
                                [years_for_pivot_table, metric_order_final_table], names=['year', 'metric']
                            )
                            total_row_df_for_concat = total_row_df_for_concat.reindex(columns=expected_cols_total_row).fillna(0)
                            total_row_df_for_concat = total_row_df_for_concat.sort_index(axis=1, level=[0,1])


                        if pivot_final_table.empty and total_row_df_for_concat.empty:
                            st.info("No detailed allocation data to display in the summary table.")
                            pivot_display_main = pd.DataFrame() # Ensure it is an empty df
                        elif pivot_final_table.empty:
                            pivot_display_main = total_row_df_for_concat
                        else: # pivot_final_table is not empty
                            if not total_row_df_for_concat.empty:
                                # Ensure columns match before concatenation if both exist
                                total_row_df_for_concat = total_row_df_for_concat.reindex(columns=pivot_final_table.columns).fillna(0)
                                pivot_display_main = pd.concat([pivot_final_table, total_row_df_for_concat])
                            else: # Only pivot_final_table exists
                                pivot_display_main = pivot_final_table


                        if not pivot_display_main.empty:
                            pivot_display_main = pivot_display_main.fillna(0) 
                            formatter_for_table_display = {} # Renamed
                            for year_col_display, metric_col_display in pivot_display_main.columns: # Renamed
                                if metric_col_display == 'volume': formatter_for_table_display[(year_col_display, metric_col_display)] = '{:,.0f} t'
                                elif metric_col_display == 'cost': formatter_for_table_display[(year_col_display, metric_col_display)] = '€{:,.2f}'
                                elif metric_col_display == 'price':
                                    # Handle 0 price for 'Total Portfolio' row specifically if it represents an average
                                    formatter_for_table_display[(year_col_display, metric_col_display)] = lambda x, r=metric_col_display: f'€{x:,.2f}/t' if pd.notna(x) and x != 0 else ('-' if (x==0 and r == 'price') else ('€0.00/t' if x==0 else '-'))
                            
                            st.dataframe(pivot_display_main.style.format(formatter_for_table_display, na_rep="-"), use_container_width=True)
                        
                    except Exception as e_table: # Renamed
                        st.error(f"Could not create or display the detailed allocation table: {e_table}")
                        st.error(f"Traceback: {traceback.format_exc()}")
                        pivot_display_main = pd.DataFrame() # Ensure it's an empty df for download check
                else: # This case means both details_df_main and summary_alloc were empty
                    st.info("No allocation data (neither details nor summary) to display in the table.")
                    pivot_display_main = pd.DataFrame() 

                if not pivot_display_main.empty:
                    csv_df_for_download = pivot_display_main.copy() # Renamed
                    # Flatten MultiIndex columns for CSV: "Year_Metric"
                    csv_df_for_download.columns = [f"{int(col[0]) if pd.notna(col[0]) and isinstance(col[0], (int, float)) else str(col[0])}_{str(col[1])}" for col in csv_df_for_download.columns.values]
                    csv_df_for_download = csv_df_for_download.reset_index() 
                    try:
                        csv_string_for_download = csv_df_for_download.to_csv(index=False, encoding='utf-8') # Renamed
                        st.markdown("---")
                        st.download_button(
                            label="Download Detailed Allocation (CSV)",
                            data=csv_string_for_download,
                            file_name=f"portfolio_allocation_{datetime.date.today()}.csv",
                            mime='text/csv',
                            key='download-csv-button-unique' # Unique key
                        )
                    except Exception as e_csv_download: # Renamed
                        st.error(f"Error preparing CSV file for download: {e_csv_download}")

            except ValueError as e_val_alloc: # Renamed
                st.error(f"ValueError during portfolio allocation or data preparation: {e_val_alloc}")
                # st.error(f"Traceback: {traceback.format_exc()}")
            except KeyError as e_key_alloc: # Renamed
                st.error(f"KeyError during portfolio allocation: Missing key '{e_key_alloc}'. This often means data required for allocation is unexpectedly missing after loading or selection. Check selections and CSV format.")
                # st.error(f"Traceback: {traceback.format_exc()}")
            except Exception as e_alloc_main: # Renamed
                st.error(f"An unexpected error occurred in the main calculation/display logic: {e_alloc_main}")
                st.error(f"Traceback: {traceback.format_exc()}")
        else:
            st.error("⚠️ Critical settings are missing or invalid for portfolio allocation. Please ensure data is loaded and all sidebar settings under 'Portfolio Settings' are configured correctly. You might need to re-upload data or adjust sliders.")

# --- Footer ---
st.markdown("---")
try:
    zurich_tz_footer = pytz.timezone('Europe/Zurich') # Renamed
    now_zurich_footer = datetime.datetime.now(zurich_tz_footer) # Renamed
    st.caption(f"Report generated: {now_zurich_footer.strftime('%Y-%m-%d %H:%M:%S %Z')}")
except Exception as e_footer: # Renamed
    now_local_footer = datetime.datetime.now() # Renamed
    st.caption(f"Report generated: {now_local_footer.strftime('%Y-%m-%d %H:%M:%S')} (Timezone: Server Local / Error: {e_footer})")
