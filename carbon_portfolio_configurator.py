# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import copy

# ==================================
# Configuration & Theming
# ==================================
# (Keep config and CSS from previous response)
st.set_page_config(layout="wide")
css = """
<style>
    /* Main App background */
    .stApp { /* background-color: #E8F5E9; */ }
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #DCEDC8; padding-top: 2rem; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stMarkdown p { color: #1B5E20; }
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] { background: #A5D6A7; }
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div:nth-child(3) { background-color: #388E3C; }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] { background-color: #66BB6A; color: white; }
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] span:first-child { background-color: #A5D6A7 !important; border-color: #388E3C !important; }
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] span:first-child[aria-checked="true"] { background-color: #388E3C !important; }
    /* Main content Margins */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; padding-left: 5rem; padding-right: 5rem; max-width: 100%; }
    /* Main Content Titles */
    h1, h2, h3, h4, h5, h6 { color: #1B5E20; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)


# ==================================
# Helper Functions (Allocation Logic)
# ==================================

# --- Main Orchestration Function for a Single Year ---
def allocate_portfolio_for_year(
    year: int,
    start_year_portfolio: int,
    end_year_portfolio: int,
    constraint_type: str,
    target_value: float,
    selected_df_all_years: pd.DataFrame,
    selected_projects_list: list,
    target_removal_percent_end_year: float,
    transition_speed: int,
    category_split: dict,
    favorite_projects: list,
    priority_boost_percent: int = 10
) -> (dict, float, float, float, float, list):
    """ Allocates projects for a single year based on constraints. Uses SIMPLE Budget logic. """
    year_str = str(year)
    year_portfolio = {}
    broken_rules_year = []
    price_col = f"price {year}"
    volume_col = f"available volume {year}"
    min_allocation_chunk = 1 # Minimum volume for inclusion pass

    if target_value <= 0:
        rule_msg = f"Target {constraint_type.lower()} for {year} is zero or less."
        return {}, 0, 0, 0, 0.0, [rule_msg]

    # --- Prepare Data for this Specific Year ---
    year_data = selected_df_all_years[
        selected_df_all_years['project name'].isin(selected_projects_list) &
        selected_df_all_years[price_col].notna() & (selected_df_all_years[price_col] > 0) &
        selected_df_all_years[volume_col].notna() & (selected_df_all_years[volume_col] >= min_allocation_chunk)
    ][[ 'project name', 'project type', price_col, volume_col] + (['priority'] if 'priority' in selected_df_all_years.columns else [])].copy()

    if 'priority' not in year_data.columns: year_data['priority'] = -1
    working_df = year_data.copy()
    if favorite_projects:
        working_df['priority'] = working_df.apply(
            lambda row: min(row['priority'] + priority_boost_percent, 100) if row['project name'] in favorite_projects and pd.notna(row['priority']) else row['priority'], axis=1)
    working_df['priority'] = working_df['priority'].fillna(-1)
    working_df.rename(columns={price_col: 'price', volume_col: 'available_volume'}, inplace=True)
    usable_projects_df = working_df.sort_values(by=['priority', 'price'], ascending=[False, True], na_position='last')

    if usable_projects_df.empty:
        rule = f"Warning {year}: No usable projects found for allocation (check selection, price/volume data for year {year})."
        return {}, 0, 0, 0, 0.0, [rule]

    # --- Calculate Target Removal % for THIS year (for reporting only in Budget mode) ---
    # (Keep the calculation using transition speed as before)
    start_removal_pct = 0.10
    end_removal_pct = target_removal_percent_end_year
    total_years_duration = end_year_portfolio - start_year_portfolio
    progress = 1.0 if total_years_duration <= 0 else max(0, min(1, (year - start_year_portfolio) / total_years_duration))
    exponent = 0.1 + (11 - transition_speed) * 0.2
    progress_factor = progress ** exponent
    target_removal_pct_year = start_removal_pct + (end_removal_pct - start_removal_pct) * progress_factor
    min_target_pct = min(start_removal_pct, end_removal_pct); max_target_pct = max(start_removal_pct, end_removal_pct)
    target_removal_pct_year = max(min_target_pct, min(max_target_pct, target_removal_pct_year))
    # Calculate category targets (used by Volume mode, reported by Budget mode)
    target_tech_pct_year = target_removal_pct_year * category_split.get('technical removal', 0)
    target_nat_pct_year = target_removal_pct_year * category_split.get('natural removal', 0)
    target_red_pct_year = 1.0 - target_tech_pct_year - target_nat_pct_year
    target_percentages = {'reduction': target_red_pct_year, 'technical removal': target_tech_pct_year, 'natural removal': target_nat_pct_year}

    # =============================================
    # --- Allocation based on Constraint Type ---
    # =============================================
    year_portfolio = {} # Final result

    # --- VOLUME MODE (Logic remains same) ---
    if constraint_type == 'Volume':
        # (Keep Volume mode logic exactly as in the previous response)
        target_total_volume = target_value
        category_target_volumes = {cat: target_total_volume * pct for cat, pct in target_percentages.items()}
        category_allocated_volumes = {cat: 0 for cat in target_percentages}
        temp_allocations = {}

        for _, row in usable_projects_df.iterrows():
            proj_name = row['project name']; proj_type = row['project type']
            price = row['price']; available = row['available_volume']; prio = row['priority']
            target_vol_for_type = category_target_volumes.get(proj_type, 0)
            allocated_vol_for_type = category_allocated_volumes.get(proj_type, 0)
            volume_needed = target_vol_for_type - allocated_vol_for_type
            if volume_needed <= 0: continue
            vol_to_take = min(volume_needed, available); vol_to_take = int(vol_to_take)
            if vol_to_take > 0:
                proj_details = temp_allocations.setdefault(proj_name, {'volume': 0, 'price': price, 'type': proj_type, 'priority_applied': prio if prio != -1 else None})
                proj_details['volume'] += vol_to_take
                category_allocated_volumes[proj_type] += vol_to_take

        initial_total_volume = sum(v['volume'] for v in temp_allocations.values())
        if initial_total_volume > 0 and target_total_volume > 0:
            scale_factor = target_total_volume / initial_total_volume
            final_scaled_volume = 0
            scaled_portfolio = {}
            for proj_name, data in temp_allocations.items():
                max_available = usable_projects_df.loc[usable_projects_df['project name'] == proj_name, 'available_volume'].iloc[0]
                scaled_vol = data['volume'] * scale_factor
                final_vol = min(int(round(scaled_vol)), int(max_available))
                if final_vol > 0:
                    scaled_portfolio[proj_name] = data.copy(); scaled_portfolio[proj_name]['volume'] = final_vol
                    final_scaled_volume += final_vol
            volume_diff = target_total_volume - final_scaled_volume
            if abs(volume_diff) > max(1, target_total_volume * 0.005): broken_rules_year.append(f"Warning {year}: Final volume ({final_scaled_volume:,.0f}) differs from target ({target_total_volume:,.0f}) due to availability caps after scaling.")
            year_portfolio = scaled_portfolio
        else: year_portfolio = {}

    # --- BUDGET MODE (SIMPLE IMPLEMENTATION) ---
    elif constraint_type == 'Budget':
        target_budget = target_value
        remaining_budget = target_budget
        budget_tolerance = 0.01 # Stop if budget less than 1 cent
        # Keep track of remaining available volume for each project
        project_availability = usable_projects_df.set_index('project name')['available_volume'].to_dict()
        allocated_portfolio = {} # project_name -> {volume: v, ...}

        # --- Main Allocation Loop (Simple Priority/Price Driven) ---
        for _, row in usable_projects_df.iterrows():
            if remaining_budget <= budget_tolerance:
                break # Stop if budget effectively zero

            proj_name = row['project name']
            price = row['price']
            proj_type = row['project type']
            prio = row['priority']
            remaining_avail = project_availability.get(proj_name, 0)

            if price <= 0 or remaining_avail <= 0: # Skip if no price or no available volume left
                continue

            # Calculate max volume affordable and possible
            vol_affordable = math.floor(remaining_budget / price)
            vol_to_allocate = int(min(vol_affordable, remaining_avail))

            if vol_to_allocate > 0:
                # Allocate this volume
                cost = vol_to_allocate * price
                proj_details = allocated_portfolio.setdefault(proj_name,
                     {'volume': 0, 'price': price, 'type': proj_type, 'priority_applied': prio if prio != -1 else None})
                proj_details['volume'] += vol_to_allocate
                remaining_budget -= cost
                project_availability[proj_name] -= vol_to_allocate # Decrease remaining available


        # --- Inclusion Pass (Heuristic - Same as before) ---
        if remaining_budget > budget_tolerance: # Only if some budget left
            projects_allocated = set(allocated_portfolio.keys())
            potential_inclusion_df = usable_projects_df[
                ~usable_projects_df['project name'].isin(projects_allocated) &
                usable_projects_df['project name'].isin(selected_projects_list)
            ].copy()

            if not potential_inclusion_df.empty:
                inclusion_candidates_sorted = potential_inclusion_df.sort_values(by=['priority', 'price'], ascending=[False, True])
                for _, row in inclusion_candidates_sorted.iterrows():
                    proj_name = row['project name']; price = row['price']
                    # Check against *original* available volume for the inclusion pass
                    original_available = row['available_volume']

                    if remaining_budget >= (price * min_allocation_chunk) and original_available >= min_allocation_chunk:
                         volume_to_allocate = min_allocation_chunk
                         cost = volume_to_allocate * price
                         # Check if project already exists from main pass (shouldn't based on filter, but safety check)
                         proj_details = allocated_portfolio.setdefault(proj_name, {'volume': 0, 'price': price, 'type': row['project type'], 'priority_applied': row['priority'] if row['priority'] != -1 else None})
                         # Only add if volume is still zero (ensure it wasn't added in main pass somehow)
                         if proj_details['volume'] == 0:
                              proj_details['volume'] += volume_to_allocate
                              remaining_budget -= cost
                              broken_rules_year.append(f"Info {year}: Allocated minimal volume ({volume_to_allocate}t) to '{proj_name}' for inclusion.")
                         if remaining_budget <= budget_tolerance: break # Stop if budget gone

        year_portfolio = allocated_portfolio

    # --- Calculate Final Summaries (Common for both modes) ---
    total_allocated_volume_year = sum(v['volume'] for v in year_portfolio.values())
    total_cost_year = sum(v['volume'] * v.get('price', 0) for v in year_portfolio.values())
    avg_price_year = (total_cost_year / total_allocated_volume_year) if total_allocated_volume_year > 0 else 0
    removal_volume = sum(v['volume'] for v in year_portfolio.values() if v.get('type') in ['technical removal', 'natural removal'])
    resulting_removal_vol_pct = (removal_volume / total_allocated_volume_year * 100) if total_allocated_volume_year > 0 else 0.0

    # --- Add final warnings/info ---
    target_budget = target_value if constraint_type == 'Budget' else 0 # Only relevant for budget mode checks

    if total_allocated_volume_year == 0 and target_value > 0:
        already_warned = any("No usable projects found" in rule for rule in broken_rules_year)
        if not already_warned: broken_rules_year.append(f"Warning {year}: No volume allocated despite non-zero target. Check budget/prices/availability.")

    # Budget specific checks/info
    if constraint_type == 'Budget':
        if total_cost_year > target_budget + budget_tolerance:
            budget_overrun = total_cost_year - target_budget
            rule = f"Error {year}: Budget constraint violated! Allocated cost ({total_cost_year:,.2f}) significantly exceeds target budget ({target_budget:,.2f}) by {budget_overrun:,.2f}."
            broken_rules_year.append(rule)
        else: # Only add comparison info if budget wasn't violated
            target_removal_pct_display = target_removal_pct_year * 100
            diff = abs(resulting_removal_vol_pct - target_removal_pct_display)
            if diff > 2.0: # Report if diff > 2%
                rule = (f"Info {year}: Target removal volume was {target_removal_pct_display:.1f}%, achieved {resulting_removal_vol_pct:.1f}%. (Budget mode allocates by priority/cost within total budget).")
                broken_rules_year.append(rule)
            # Budget utilization info
            if target_budget > 0:
                utilization = (total_cost_year / target_budget) * 100 if target_budget > 0 else 0
                # Add info if utilization is very low, e.g. < 50% and budget was significant
                if utilization < 50 and target_budget > 1000: # Thresholds adjustable
                     rule = f"Info {year}: Budget utilization: {utilization:.1f}% ({total_cost_year:,.2f} / {target_budget:,.2f}). Remaining projects may be too expensive or budget already spent on priorities."
                     broken_rules_year.append(rule)


    broken_rules_year = sorted(list(set(broken_rules_year)))
    return year_portfolio, total_allocated_volume_year, total_cost_year, avg_price_year, resulting_removal_vol_pct, broken_rules_year


# ==================================
# Streamlit App Layout & Logic
# ==================================
# (Sidebar and main app structure remain the same as the previous response)
# ... [Paste the Streamlit UI code from the previous response here] ...
# The call to allocate_portfolio_for_year should work correctly with this simplified logic.

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 1. Load Data")
    df_upload = st.file_uploader("Upload Project Data CSV", type="csv", key="uploader")

    # Initialize session state (important for dynamic updates)
    if 'working_data_full' not in st.session_state: st.session_state.working_data_full = None
    if 'selected_years' not in st.session_state: st.session_state.selected_years = []
    if 'selected_projects' not in st.session_state: st.session_state.selected_projects = []
    if 'project_names' not in st.session_state: st.session_state.project_names = []
    if 'favorite_projects' not in st.session_state: st.session_state.favorite_projects = []
    if 'actual_start_year' not in st.session_state: st.session_state.actual_start_year = None
    if 'actual_end_year' not in st.session_state: st.session_state.actual_end_year = None
    if 'available_years_in_data' not in st.session_state: st.session_state.available_years_in_data = []
    # Add defaults for sliders if not set
    if 'constraint_type' not in st.session_state: st.session_state.constraint_type = 'Volume'
    if 'removal_target_end_year' not in st.session_state: st.session_state.removal_target_end_year = 0.8
    if 'transition_speed' not in st.session_state: st.session_state.transition_speed = 5
    if 'removal_preference' not in st.session_state: st.session_state.removal_preference = 5


    if df_upload:
        try:
            # --- Data Loading and Preparation ---
            @st.cache_data # Cache the data loading and initial processing
            def load_and_prepare_data(uploaded_file):
                # (Keep the robust loading/parsing from previous version)
                data = pd.read_csv(uploaded_file)
                data.columns = data.columns.str.lower().str.strip().str.replace(' ', '_')
                core_cols_std = ['project_name', 'project_type']
                numeric_prefixes_std = ['price_', 'available_volume_', 'priority']
                missing_essential = [col for col in core_cols_std if col not in data.columns]
                if missing_essential: return None, f"Missing essential columns: {', '.join(missing_essential)}", [], []
                cols_to_convert = []
                available_years = set()
                for col in data.columns:
                    is_numeric_candidate = False
                    if col == 'priority': is_numeric_candidate = True
                    else:
                        for prefix in numeric_prefixes_std:
                            if col.startswith(prefix) and col[len(prefix):].isdigit():
                                is_numeric_candidate = True; available_years.add(int(col[len(prefix):])); break
                    if is_numeric_candidate: cols_to_convert.append(col)
                for col in list(set(cols_to_convert)):
                    if col in data.columns: data[col] = pd.to_numeric(data[col], errors='coerce')
                available_years = sorted(list(available_years))
                if not available_years: return None, "No valid year data columns found.", [], []
                if 'project_type' in data.columns:
                    data['project_type'] = data['project_type'].str.lower().str.strip()
                    valid_types = ['technical removal', 'natural removal', 'reduction']
                    # Store invalid types before filtering for warning message persistence
                    invalid_types_df = data[~data['project_type'].isin(valid_types)]
                    invalid_types_found = invalid_types_df['project_type'].unique() if not invalid_types_df.empty else []
                    data = data[data['project_type'].isin(valid_types)] # Filter out invalid types
                else: invalid_types_found = [] # No project_type column
                column_mapping_to_display = {col: col.replace('_', ' ') for col in data.columns}
                data.rename(columns=column_mapping_to_display, inplace=True)
                project_names_list = sorted(data['project name'].unique().tolist())
                return data, None, available_years, project_names_list, invalid_types_found # Return invalid types

            # --- Execute data loading ---
            data, error_msg, available_years_in_data, project_names_list, invalid_types_found = load_and_prepare_data(df_upload)

            # Display warning for invalid types outside cache function
            if invalid_types_found:
                 st.sidebar.warning(f"Invalid project types ignored: {', '.join(invalid_types_found)}")


            if error_msg:
                st.sidebar.error(error_msg)
                st.session_state.working_data_full = None # Reset state on error
            else:
                st.session_state.project_names = project_names_list
                st.session_state.available_years_in_data = available_years_in_data
                st.session_state.working_data_full = data # Store the full loaded data

                # --- Year Selection ---
                st.markdown("## 2. Portfolio Settings")
                min_year = min(available_years_in_data)
                max_year_possible = max(available_years_in_data)
                years_max_slider = min(10, max_year_possible - min_year + 1)
                years_max_slider = max(1, years_max_slider)

                years_to_plan = st.slider(f"Years to Plan (from {min_year})", 1, years_max_slider, min(3, years_max_slider), key='years_slider_sidebar')
                start_year = min_year
                end_year = start_year + years_to_plan - 1
                selected_years_range = list(range(start_year, end_year + 1))

                # Determine actual available years in range
                actual_years_present = []
                for year in selected_years_range:
                    price_col = f"price {year}"; vol_col = f"available volume {year}"
                    if price_col in data.columns and vol_col in data.columns: actual_years_present.append(year)

                st.session_state.selected_years = actual_years_present
                if not st.session_state.selected_years:
                    st.sidebar.error(f"No data found for years in range {start_year}-{end_year}.")
                    st.session_state.working_data_full = None # Reset if no valid years
                else:
                    st.session_state.actual_start_year = min(st.session_state.selected_years)
                    st.session_state.actual_end_year = max(st.session_state.selected_years)

                    # --- Constraint & Targets ---
                    # Use st.session_state.get to handle initial load before widgets assigned
                    st.session_state.constraint_type = st.radio("Constraint Type:", ('Volume', 'Budget'), index=['Volume', 'Budget'].index(st.session_state.get('constraint_type', 'Volume')), key='constraint_type_sidebar', horizontal=True)
                    constraint_type = st.session_state.constraint_type # Local variable for immediate use

                    annual_targets = {}
                    if constraint_type == 'Volume':
                        master_target = st.number_input("Target Volume (per year):", min_value=0, step=100, value=1000, key='master_volume_sidebar')
                        tgt_fmt = "%d"
                    else: # Budget
                        master_target = st.number_input("Target Budget (€ per year):", min_value=0.0, step=1000.0, value=100000.0, format="%.2f", key='master_budget_sidebar') # Increased default
                        tgt_fmt = "%.2f"

                    with st.expander("Adjust Yearly Targets (Optional)"):
                        for year in st.session_state.selected_years:
                            # Get default value potentially stored from previous run
                            default_val = st.session_state.get('annual_targets', {}).get(year, master_target)
                            annual_targets[year] = st.number_input(f"Target {year}", value=default_val, format=tgt_fmt, key=f"target_{year}_sidebar")

                    st.session_state.annual_targets = annual_targets # Store updated targets

                    # --- Removal & Transition ---
                    st.markdown("### Removal & Transition")
                    removal_target_percent_slider = st.slider(f"Target Removal Vol % for **Year {st.session_state.actual_end_year}**", 0, 100, int(st.session_state.get('removal_target_end_year', 0.8)*100), help="Target % of *volume* from Removals in the final year. Starts at 10%.", key='removal_perc_slider_sidebar')
                    st.session_state.removal_target_end_year = removal_target_percent_slider / 100.0

                    col_t, col_p = st.columns(2)
                    with col_t: st.session_state.transition_speed = st.slider("Transition Speed", 1, 10, st.session_state.get('transition_speed', 5), help="Speed of ramping up removal % (1=Slow, 10=Fast)", key='transition_speed_slider_sidebar')
                    with col_p: st.session_state.removal_preference = st.slider("Technical Removal Preference", 1, 10, st.session_state.get('removal_preference', 5), help="Focus within removal target (1=Natural, 10=Technical)", key='removal_pref_slider_sidebar')

                    # --- Project Selection ---
                    st.markdown("## 3. Select Projects")
                    # Use session state for default selection to preserve it across reruns
                    st.session_state.selected_projects = st.multiselect("Select projects for portfolio:", st.session_state.project_names, default=st.session_state.get('selected_projects', st.session_state.project_names), key='project_selector_sidebar')
                    available_for_boost = [p for p in st.session_state.project_names if p in st.session_state.selected_projects]
                    # Preserve favorite selection using session state default
                    st.session_state.favorite_projects = st.session_state.get('favorite_projects', [])
                    if 'priority' in data.columns:
                        if available_for_boost: st.session_state.favorite_projects = st.multiselect("Favorite projects (+10% priority boost):", available_for_boost, default=st.session_state.favorite_projects, key='favorite_selector_sidebar')
                    else: st.info("No 'priority' column; boosting disabled.")

        except Exception as e:
            st.sidebar.error(f"Error processing file: {e}")
            st.exception(e)
            st.session_state.working_data_full = None

    else: # No file uploaded
         st.session_state.working_data_full = None
         st.session_state.selected_years = []
         st.session_state.selected_projects = []


# ==================================
# Main Page Content (Results)
# ==================================

st.title("Carbon Portfolio Builder")

# Check if ready to allocate (use .get for safety on first run)
if st.session_state.get('working_data_full') is None or not st.session_state.get('selected_projects'):
    st.info("👋 Welcome! Please upload data and select projects using the sidebar to build your portfolio.")
else:
    # --- Get data from session state for allocation ---
    # Ensure all needed keys exist before proceeding
    required_state_keys = ['working_data_full', 'selected_projects', 'selected_years', 'annual_targets', 'constraint_type', 'removal_target_end_year', 'transition_speed', 'removal_preference', 'favorite_projects', 'actual_start_year', 'actual_end_year']
    if all(key in st.session_state for key in required_state_keys):

        working_data_full = st.session_state.working_data_full
        selected_projects = st.session_state.selected_projects
        selected_years = st.session_state.selected_years
        annual_targets = st.session_state.annual_targets
        constraint_type = st.session_state.constraint_type
        removal_target_end_year = st.session_state.removal_target_end_year
        transition_speed = st.session_state.transition_speed
        removal_preference = st.session_state.removal_preference
        favorite_projects = st.session_state.favorite_projects
        start_year = st.session_state.actual_start_year
        end_year = st.session_state.actual_end_year

        with st.spinner("Calculating portfolio allocation..."):

            # --- Calculate Category Split based on preference ---
            tech_pref_normalized = (removal_preference - 1) / 9.0
            category_split = {'technical removal': tech_pref_normalized, 'natural removal': 1.0 - tech_pref_normalized}

            # --- Main Allocation Loop ---
            portfolio = {}
            yearly_summary_list = []
            all_broken_rules = []
            final_removal_percentages = {}

            for year_alloc in selected_years:
                target_value_year = annual_targets.get(year_alloc, 0)
                selected_df_all_years_subset = working_data_full[working_data_full['project name'].isin(selected_projects)].copy()

                (year_portfolio_result, total_allocated_volume_year, total_cost_year, avg_price_year,
                 resulting_removal_vol_pct, broken_rules_year) = allocate_portfolio_for_year(
                    year=year_alloc, start_year_portfolio=start_year, end_year_portfolio=end_year,
                    constraint_type=constraint_type, target_value=target_value_year,
                    selected_df_all_years=selected_df_all_years_subset, selected_projects_list = selected_projects,
                    target_removal_percent_end_year=removal_target_end_year, transition_speed=transition_speed,
                    category_split=category_split, favorite_projects=favorite_projects, priority_boost_percent=10 )

                portfolio[year_alloc] = year_portfolio_result
                final_removal_percentages[year_alloc] = resulting_removal_vol_pct
                yearly_summary_list.append({'Year': year_alloc, f'Target {constraint_type}': target_value_year, 'Allocated Volume': total_allocated_volume_year, 'Total Cost': total_cost_year, 'Avg. Price': avg_price_year, 'Actual Removal Vol %': resulting_removal_vol_pct })
                all_broken_rules.extend(broken_rules_year)

            # --- Display Results ---
            st.header("📈 Portfolio Allocation Results")
            st.markdown("---")
            # (Warnings/Errors display remains same)
            if all_broken_rules:
                unique_rules = sorted(list(set(all_broken_rules)))
                error_rules = [r for r in unique_rules if "Error" in r]
                warning_rules = [r for r in unique_rules if "Warning" in r or "Info" in r]
                if error_rules:
                     st.error("Errors during allocation:")
                     for rule in error_rules: st.error(f"- {rule}")
                if warning_rules:
                     st.warning("Warnings/Info during allocation:")
                     for rule in warning_rules: st.warning(f"- {rule}")
                st.markdown("---")

            # (Summary Table display remains same)
            if yearly_summary_list:
                summary_display_df = pd.DataFrame(yearly_summary_list)
                st.dataframe(summary_display_df.style.format({f'Target {constraint_type}': '{:,.2f}' if constraint_type == 'Budget' else '{:,.0f}', 'Allocated Volume': '{:,.0f}', 'Total Cost': '€{:,.2f}', 'Avg. Price': '€{:,.2f}', 'Actual Removal Vol %': '{:.1f}%'}), hide_index=True, use_container_width=True)
            else: st.info("No allocation results to display in summary.")

            # (Detailed Allocation Tabs display remains same)
            if portfolio and any(portfolio.values()):
                st.subheader("📄 Detailed Allocation per Year")
                years_with_allocations = [yr for yr in selected_years if yr in portfolio and portfolio[yr]]
                if years_with_allocations:
                    tab_list = st.tabs([str(year) for year in years_with_allocations])
                    for i, year_tab in enumerate(years_with_allocations):
                        with tab_list[i]:
                            #...(rest of tab display code)...
                            year_data = portfolio[year_tab]
                            alloc_list = []
                            for proj, details in year_data.items():
                                 alloc_list.append({ 'Project Name': proj, 'Type': details.get('type', 'N/A'), 'Volume': details.get('volume', 0), 'Price': details.get('price', 0), 'Cost': details.get('volume', 0) * details.get('price', 0), 'Priority Applied': details.get('priority_applied', 'N/A') })
                            year_df = pd.DataFrame(alloc_list).sort_values(by='Project Name')
                            st.dataframe(year_df.style.format({ 'Volume': '{:,.0f}', 'Price': '€{:,.2f}', 'Cost': '€{:,.2f}', 'Priority Applied': '{}' }), hide_index=True, use_container_width=True)
                else: st.info("No projects allocated in any selected year.")


            # --- Visualization Code Block ---
            # (Visualization code remains the same)
            st.header("📊 Portfolio Analysis & Visualization")
            st.markdown("---")
            all_types_in_portfolio = set()
            portfolio_data_list = []
            if portfolio:
                for year_viz, projects in portfolio.items():
                    if projects:
                        for name, info in projects.items():
                            if info.get('volume', 0) > 0:
                                all_types_in_portfolio.add(info['type'])
                                portfolio_data_list.append({ 'year': year_viz, 'project name': name, 'type': info['type'], 'volume': info['volume'], 'price': info['price'], 'cost': info['volume'] * info['price'] })

            if not portfolio_data_list:
                st.warning("No projects with positive volumes allocated. Cannot generate plots.")
            else:
                portfolio_df = pd.DataFrame(portfolio_data_list)
                # (Rest of plotting code remains the same...)
                summary_plot_df = portfolio_df.groupby(['year', 'type']).agg(volume=('volume', 'sum'), cost=('cost', 'sum')).reset_index()
                price_summary = summary_plot_df.groupby('year').agg(total_volume=('volume', 'sum'), total_cost=('cost', 'sum')).reset_index()
                price_summary['avg_price'] = price_summary.apply(lambda row: row['total_cost'] / row['total_volume'] if row['total_volume'] > 0 else 0, axis=1)
                st.subheader("Portfolio Composition & Price Over Time")
                fig_composition = make_subplots(specs=[[{"secondary_y": True}]])
                color_map = {'technical removal': '#64B5F6', 'natural removal': '#81C784', 'reduction': '#FFB74D'}
                default_color = '#B0BEC5'
                plot_type_order = ['reduction', 'natural removal', 'technical removal']
                for type_name in plot_type_order:
                    if type_name in all_types_in_portfolio:
                        df_type = summary_plot_df[summary_plot_df['type'] == type_name]
                        fig_composition.add_trace(go.Bar(x=df_type['year'], y=df_type['volume'], name=type_name.capitalize(), marker_color=color_map.get(type_name, default_color), hovertemplate='Year: %{x}<br>Type: '+type_name.capitalize()+'<br>Volume: %{y:,.0f}<extra></extra>'), secondary_y=False)
                actual_removal_df = pd.DataFrame({'year': list(final_removal_percentages.keys()), 'actual_pct': list(final_removal_percentages.values())})
                fig_composition.add_trace(go.Scatter(x=actual_removal_df['year'], y=actual_removal_df['actual_pct'], name='Actual Removal Vol %', mode='lines+markers', line=dict(color='purple', dash='dash'), marker=dict(symbol='star'), hovertemplate='Year: %{x}<br>Actual Removal: %{y:.1f}%<extra></extra>'), secondary_y=True)
                fig_composition.add_trace(go.Scatter(x=price_summary['year'], y=price_summary['avg_price'], name='Avg Price (€/t)', mode='lines+markers', marker=dict(symbol='circle', size=8), line=dict(color='#546E7A', width=3), hovertemplate='Year: %{x}<br>Avg Price: €%{y:,.2f}<extra></extra>'), secondary_y=True)
                fig_composition.update_layout(xaxis_title='Year', yaxis_title='Volume (tonnes)', yaxis2_title='Avg Price (€/t) / Actual Removal %', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), barmode='stack', template="plotly_white", yaxis=dict(rangemode='tozero'), yaxis2=dict(rangemode='tozero'), hovermode="x unified")
                fig_composition.update_xaxes(tickmode='array', tickvals=selected_years, dtick=1)
                st.plotly_chart(fig_composition, use_container_width=True)

                st.subheader("Total Allocation Overview (Across All Years)")
                if not portfolio_df.empty:
                    df_summed_sunburst = portfolio_df.groupby(['type', 'project name']).agg({'volume': 'sum'}).reset_index()
                    df_summed_sunburst = df_summed_sunburst[df_summed_sunburst['volume'] > 0]
                    if not df_summed_sunburst.empty:
                        df_summed_sunburst['display_name'] = df_summed_sunburst['project name']
                        fig_sunburst = px.sunburst(df_summed_sunburst, path=['type', 'display_name'], values='volume', color='type', color_discrete_map=color_map, title=None, branchvalues="total")
                        fig_sunburst.update_traces(textinfo='label+percent parent', insidetextorientation='radial', hovertemplate='<b>%{label}</b><br>Volume: %{value:,.0f}<br>Parent Pct: %{percentParent:.1%}<extra></extra>')
                        fig_sunburst.update_layout(margin=dict(t=20, l=0, r=0, b=0))
                        st.plotly_chart(fig_sunburst, use_container_width=True)
                    else: st.info("No projects with positive total volume for sunburst chart.")

                st.subheader("Detailed Project Allocation Over Time")
                if not portfolio_df.empty:
                    fig_grouped_projects = go.Figure()
                    unique_projects = portfolio_df['project name'].unique()
                    project_color_scale = px.colors.qualitative.Plotly + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
                    project_color_map = {proj: project_color_scale[i % len(project_color_scale)] for i, proj in enumerate(unique_projects)}
                    for type_name in plot_type_order:
                         type_projects_df = portfolio_df[portfolio_df['type'] == type_name]
                         projects_in_type = sorted(type_projects_df['project name'].unique())
                         for project in projects_in_type:
                              project_data = type_projects_df[type_projects_df['project name'] == project]
                              fig_grouped_projects.add_trace(go.Bar(x=project_data['year'], y=project_data['volume'], name=project, marker_color=project_color_map.get(project, default_color), legendgroup=type_name, legendgrouptitle_text=type_name.capitalize(), hovertemplate = 'Year: %{x}<br>Project: '+project+'<br>Volume: %{y:,.0f}<extra></extra>'))
                    fig_grouped_projects.update_layout(xaxis_title='Year', yaxis_title='Volume (tonnes)', legend_title='Projects by Type', legend=dict(tracegroupgap=10, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), barmode='stack', template="plotly_white", yaxis=dict(rangemode='tozero'), xaxis=dict(tickmode='array', tickvals=selected_years, dtick=1))
                    st.plotly_chart(fig_grouped_projects, use_container_width=True)

                if not portfolio_df.empty:
                    with st.expander("Show Detailed Allocation Data Table"):
                        display_portfolio_df = portfolio_df.copy()
                        display_portfolio_df['volume'] = display_portfolio_df['volume'].map('{:,.0f}'.format)
                        display_portfolio_df['price'] = display_portfolio_df['price'].map('€{:,.2f}'.format)
                        display_portfolio_df['cost'] = display_portfolio_df['cost'].map('€{:,.2f}'.format)
                        st.dataframe(display_portfolio_df[['year', 'project name', 'type', 'volume', 'price', 'cost']].sort_values(by=['year', 'project name']), hide_index=True, use_container_width=True)

    else:
         # This case handles when essential keys are missing from session_state after upload
         st.warning("Session state incomplete. Please ensure data is loaded and settings are selected.")
