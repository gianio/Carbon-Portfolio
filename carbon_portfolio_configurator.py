import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np # Added for potential NaN checks if needed

# --- Define a nice green color palette ---
primary_green = "#D7F0C8" # Light Green
secondary_green = "#D7F0C8" # Medium Green
background_green = "#D7F0C8" # Very Light Green
text_green = "#33691E" # Dark Green
accent_green = "#D7F0C8" # Lime Green

# --- Streamlit App Configuration ---
st.set_page_config(layout="centered") # Optional: Use wider layout
st.markdown(f"""
    <style>
    body {{
        background-color: {background_green};
        font-family: Calibri, sans-serif;
        font-size: 20.8px; /* 16px * 1.3 */
        color: {text_green};
    }}
    .stApp {{
        background-color: {background_green};
    }}
    .stApp > header {{
        margin-bottom: 13px; /* 10px * 1.3 */
    }}
    h1 {{
        color: {secondary_green};
        font-family: Calibri, sans-serif;
        font-size: 39px; /* 30px * 1.3 */
    }}
    h2 {{
        border-bottom: 2.6px solid {accent_green}; /* 2px * 1.3 */
        padding-bottom: 6.5px; /* 5px * 1.3 */
        margin-top: 26px; /* 20px * 1.3 */
        font-family: Calibri, sans-serif;
        font-size: 31.2px; /* 24px * 1.3 */
        color: {secondary_green};
    }}
    h3 {{
        font-family: Calibri, sans-serif;
        font-size: 26px; /* 20px * 1.3 */
        color: {secondary_green};
    }}
    p, div, stText, stMarkdown, stCaption, stNumberInput label, stSlider label, stFileUploader label, stMultiSelect label, stRadio label, stCheckbox label {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important; /* 16px * 1.3 */
        color: {text_green} !important;
    }}
    .streamlit-expander {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
        border-color: {accent_green} !important;
    }}
    .streamlit-expander-content {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
    }}
    .stButton > button {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
        border-color: {secondary_green} !important;
        background-color: {background_green} !important;
        &:hover {{
            background-color: {primary_green} !important;
            color: white !important;
        }}
    }}
    .stSlider > div[data-baseweb="slider"] > div[role="slider"]::before {{
        background-color: {primary_green} !important;
    }}
    .stSlider > div[data-baseweb="slider"] > div[role="slider"] > span {{
        background-color: {secondary_green} !important;
        border-color: {secondary_green} !important;
    }}
    .stNumberInput > div > div > input {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
        border-color: {accent_green} !important;
    }}
    .stSelectbox > div > div > div > button {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
        border-color: {accent_green} !important;
        background-color: {background_green} !important;
    }}
    .stMultiSelect > div > div > div > button {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
        border-color: {accent_green} !important;
        background-color: {background_green} !important;
    }}
    .stRadio > label {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
    }}
    .stCheckbox > label {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
    }}
    .stFileUploader > div > div:first-child > div:first-child > label {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
        border-color: {accent_green} !important;
        background-color: {background_green} !important;
    }}
    .stDataFrame {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
        border-color: {accent_green} !important;
    }}
    .stDataFrame tr th {{
        background-color: {accent_green} !important;
        color: {text_green} !important;
    }}
    .stMetric {{
        background-color: {background_green} !important;
        border: 1px solid {accent_green} !important;
        padding: 15px !important;
        border-radius: 5px !important;
    }}
    .stMetricLabel {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {secondary_green} !important;
    }}
    .stMetricValue {{
        font-family: Calibri, sans-serif !important;
        font-size: 26px !important;
        color: {text_green} !important;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("🌱 Multi-Year Carbon Portfolio Builder")

# --- Data Input ---
df_upload = st.file_uploader("Upload project CSV", type="csv")

# --- Main Logic (Conditional on CSV Upload) ---
if df_upload:
    data = pd.read_csv(df_upload)
    # Basic check for essential columns (can be expanded)
    required_cols = ['project name', 'project type']
    if not all(col in data.columns for col in required_cols):
        st.error(f"CSV must contain at least the following columns: {', '.join(required_cols)}")
        st.stop() # Stop execution if essential columns are missing

    # --- Project Overview Section ---
    st.subheader("Project Overview")
    years = st.number_input("How many years should the portfolio span?", min_value=1, max_value=20, value=6, step=1)
    start_year = 2025 # Define start year explicitly (Adjust if needed)
    end_year = start_year + years - 1
    selected_years = list(range(start_year, end_year + 1))

    overview = data.copy()
    price_cols = [f"price {year}" for year in selected_years if f"price {year}" in overview.columns]

    # Ensure 'priority' column exists, fill with a default if not (e.g., 50 or median)
    if 'priority' not in overview.columns:
        overview['priority'] = 50 # Assign a neutral default priority
        # st.info("No 'priority' column found in CSV. Assigning a default priority of 50 to all projects.") # Less verbose
    else:
         # Fill NaN priorities with default
        overview['priority'] = overview['priority'].fillna(50)

    # Calculate average price only if price columns exist
    if price_cols:
        # Ensure price columns are numeric, coerce errors to NaN
        for col in price_cols:
             overview[col] = pd.to_numeric(overview[col], errors='coerce')
        overview['avg_price'] = overview[price_cols].mean(axis=1)
        overview_display_cols = ['project name', 'project type']
        if 'description' in overview.columns:
             overview_display_cols.append('description')
        overview_display_cols.append('avg_price')
        st.dataframe(overview[overview_display_cols].drop_duplicates(subset=['project name']).reset_index(drop=True))
    else:
        overview_display_cols = ['project name', 'project type']
        if 'description' in overview.columns:
            overview_display_cols.append('description')
        st.dataframe(overview[overview_display_cols].drop_duplicates(subset=['project name']).reset_index(drop=True))
        st.warning(f"No price columns found for years {start_year}-{end_year} (e.g., 'price {start_year}'). Cannot calculate average price.")


    # --- Portfolio Settings Section ---
    st.subheader("Step 1: Define Portfolio Settings")
    constraint_type = st.radio("Select constraint type:", ["Volume Constrained", "Budget Constrained"], key="constraint_type", horizontal=True)
    annual_constraints = {}

    # Use columns for layout
    col1, col2 = st.columns([0.6, 0.4]) # Adjust column width ratio if needed

    st.markdown("**Annual Constraints:**")
    # Display yearly inputs in a more compact way if many years
    cols_per_year = st.columns(years if years <= 6 else 6) # Max 6 columns for inputs
    col_idx = 0
    if constraint_type == "Volume Constrained":
        st.markdown("Enter annual purchase volumes (tonnes):")
        # REMOVED global_volume input
        for year in selected_years:
            with cols_per_year[col_idx % len(cols_per_year)]:
                # Use year as default value for easier debugging if needed, otherwise 1000
                default_val = 1000
                annual_constraints[year] = st.number_input(f"{year}", min_value=0, step=100, value=default_val, key=f"vol_{year}", label_visibility="visible") # Show year label
            col_idx += 1
    else:
        st.markdown("Enter annual budget (€):")
        # REMOVED global_budget input
        for year in selected_years:
             with cols_per_year[col_idx % len(cols_per_year)]:
                # Use year*10 as default value for easier debugging if needed, otherwise 10000
                default_val = 10000
                annual_constraints[year] = st.number_input(f"{year} (€)", min_value=0, step=1000, value=default_val, key=f"bud_{year}", label_visibility="visible") # Show year label
             col_idx += 1

    
    st.markdown("**Portfolio Strategy:**")
    removal_target = st.slider(f"Target Removal % by {end_year}", 0, 100, 80, key="removal_target") / 100
    transition_speed = st.slider("Transition Speed (1=Slow, 10=Fast)", 1, 10, 5, key="transition_speed")
    # Make label clearer about slider direction
    removal_preference = st.slider("Removal Preference (1=Natural Favored, 10=Technical Favored)", 1, 10, 5, key="removal_preference")
    st.caption("Preference influences allocation order and targets in mixed portfolios.")


    # --- Project Selection and Prioritization Section ---
    st.subheader("Step 2: Select and Prioritize Projects")
    project_names = sorted(data['project name'].unique().tolist()) # Sort alphabetically
    selected_projects = st.multiselect("Select projects to include:", project_names, default=project_names, key="select_proj")
    favorite_projects = st.multiselect("Select favorite projects (+10% priority boost):", selected_projects, key="fav_proj")

    if selected_projects:
        selected_df = data[data['project name'].isin(selected_projects)].copy()

        # Apply favorite boost (ensure priority column exists from earlier check)
        if favorite_projects:
            selected_df['priority'] = selected_df.apply(
                lambda row: row['priority'] + 10 if row['project name'] in favorite_projects else row['priority'],
                axis=1
            )
            selected_df['priority'] = selected_df['priority'].clip(upper=100) # Cap priority at 100

        # Identify available project types globally selected
        global_selected_types = selected_df['project type'].unique().tolist()
        global_removal_types = [t for t in global_selected_types if t in ['technical removal', 'natural removal']]
        global_reduction_type = 'reduction' if 'reduction' in global_selected_types else None
        global_has_removals = bool(global_removal_types)
        global_has_reductions = bool(global_reduction_type)


        # --- Allocation Logic ---
        portfolio = {year: {} for year in selected_years} # Stores { project_name: { volume: v, price: p, type: t } }
        allocation_warnings = []
        
        # --- Start Year Loop ---
        for year_idx, year in enumerate(selected_years):
            year_str = str(year)
            volume_col = f"available volume {year_str}"
            price_col = f"price {year_str}"

            # Check if essential columns exist for the current year
            if volume_col not in selected_df.columns or price_col not in selected_df.columns:
                allocation_warnings.append(f"Warning for {year}: Missing '{volume_col}' or '{price_col}' column. Skipping allocation for this year.")
                portfolio[year] = {}
                continue

            # Prepare DataFrame for this year's allocation
            year_df = selected_df[['project name', 'project type', 'priority', volume_col, price_col]].copy()
            year_df.rename(columns={volume_col: 'volume', price_col: 'price'}, inplace=True)

            # Convert volume and price to numeric, handle errors/NaNs
            year_df['volume'] = pd.to_numeric(year_df['volume'], errors='coerce').fillna(0)
            year_df['price'] = pd.to_numeric(year_df['price'], errors='coerce')

            # Filter out projects with zero volume or invalid price for the year
            year_df = year_df[(year_df['volume'] > 0) & pd.notna(year_df['price'])]

            if year_df.empty:
                portfolio[year] = {}
                continue

            # --- Year Specific Calculations ---
            annual_limit = annual_constraints.get(year, 0)
            if annual_limit <= 0:
                portfolio[year] = {}
                continue # Skip year if constraint is zero

            total_allocated_volume_year = 0.0
            total_allocated_cost_year = 0.0
            allocated_projects_this_year = {}

            # Calculate target split for removals/reductions based on transition
            year_fraction = (year_idx + 1) / years
            current_removal_target_fraction = removal_target * (year_fraction ** (0.5 + 0.1 * transition_speed))

            # Determine available types FOR THIS YEAR and portfolio type
            current_year_types = year_df['project type'].unique().tolist()
            current_has_removals = any(t in ['technical removal', 'natural removal'] for t in current_year_types)
            current_has_reductions = 'reduction' in current_year_types
            current_only_removals = current_has_removals and not current_has_reductions
            current_only_reductions = not current_has_removals and current_has_reductions

            if current_only_reductions:
                current_removal_target_fraction = 0.0
            elif current_only_removals:
                 current_removal_target_fraction = 1.0
            elif not current_has_removals and not current_has_reductions:
                portfolio[year] = {}
                continue

            target_removal_alloc = annual_limit * current_removal_target_fraction
            target_reduction_alloc = annual_limit * (1.0 - current_removal_target_fraction)

            # Sort projects: High priority first, then low price
            year_df.sort_values(by=['priority', 'price'], ascending=[False, True], inplace=True)

            # Get DFs for removals and reductions for this year
            removals_df = year_df[year_df['project type'].isin(global_removal_types)].copy() if current_has_removals else pd.DataFrame()
            reductions_df = year_df[year_df['project type'] == global_reduction_type].copy() if current_has_reductions else pd.DataFrame()

            # --- PHASE 1: Targeted Allocation ---
            allocated_natural_value = 0.0 # Track allocated volume/cost for each removal type in Phase 1
            allocated_technical_value = 0.0
            allocated_reduction_value = 0.0

            # 1.a Allocate Removals (respecting preference)
            if current_has_removals and not removals_df.empty:
                removal_pref_factor = 1 / (1 + np.exp(-1.2 * (removal_preference - 5.5)))
                natural_pref_target = target_removal_alloc * (1.0 - removal_pref_factor)
                technical_pref_target = target_removal_alloc * removal_pref_factor

                if removal_pref_factor > 0.55:
                    removal_order_types = ['technical removal', 'natural removal']
                else:
                    removal_order_types = ['natural removal', 'technical removal']

                for r_type in removal_order_types:
                    if r_type not in removals_df['project type'].unique(): continue
                    type_df = removals_df[removals_df['project type'] == r_type]

                    for idx, project in type_df.iterrows():
                        if (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                           (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6): break
                        project_name = project['project name']
                        available_vol = year_df.loc[idx, 'volume']
                        price = project['price']
                        if available_vol < 1e-6: continue
                        vol_to_allocate = 0.0

                        # Define remaining limits
                        remaining_overall_limit_vol = max(0, annual_limit - total_allocated_volume_year)
                        remaining_overall_limit_cost = max(0, annual_limit - total_allocated_cost_year)
                        cost_of_unit = price if price > 0 else 0
                        affordable_vol_overall = remaining_overall_limit_cost / (cost_of_unit + 1e-9)

                        if constraint_type == "Volume Constrained":
                            if current_only_removals:
                                vol_to_allocate = min(available_vol, remaining_overall_limit_vol)
                            else:
                                if r_type == 'natural removal':
                                    remaining_pref_target_vol = max(0, natural_pref_target - allocated_natural_value)
                                else:
                                    remaining_pref_target_vol = max(0, technical_pref_target - allocated_technical_value)
                                vol_to_allocate = min(available_vol, remaining_overall_limit_vol, remaining_pref_target_vol)
                        else: # Budget Constrained
                            if current_only_removals:
                                vol_to_allocate = min(available_vol, affordable_vol_overall)
                            else:
                                if r_type == 'natural removal':
                                    remaining_pref_target_cost = max(0, natural_pref_target - allocated_natural_value)
                                    affordable_vol_pref = remaining_pref_target_cost / (cost_of_unit + 1e-9)
                                else:
                                    remaining_pref_target_cost = max(0, technical_pref_target - allocated_technical_value)
                                    affordable_vol_pref = remaining_pref_target_cost / (cost_of_unit + 1e-9)
                                vol_to_allocate = min(available_vol, affordable_vol_overall, affordable_vol_pref)

                        if vol_to_allocate < 1e-6: vol_to_allocate = 0.0
                        if vol_to_allocate > 0:
                            cost = vol_to_allocate * price
                            total_allocated_volume_year += vol_to_allocate
                            total_allocated_cost_year += cost
                            if project_name not in allocated_projects_this_year:
                                allocated_projects_this_year[project_name] = {'volume': 0.0, 'price': price, 'type': r_type}
                            allocated_projects_this_year[project_name]['volume'] += vol_to_allocate
                            year_df.loc[idx, 'volume'] -= vol_to_allocate # Decrement remaining volume
                            if r_type == 'natural removal':
                                allocated_natural_value += (cost if constraint_type == "Budget Constrained" else vol_to_allocate)
                            else:
                                allocated_technical_value += (cost if constraint_type == "Budget Constrained" else vol_to_allocate)
                    # Check limit after each type
                    if (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                       (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6): break

            # 1.b Allocate Reductions (Only if reductions exist and limit not met)
            if current_has_reductions and not reductions_df.empty:
                if not ((constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                        (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6)):
                    for idx, project in reductions_df.iterrows():
                        if (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                           (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6): break
                        project_name = project['project name']
                        available_vol = year_df.loc[idx, 'volume']
                        price = project['price']
                        if available_vol < 1e-6: continue
                        vol_to_allocate = 0.0
                        cost_of_unit = price if price > 0 else 0

                        remaining_overall_limit_vol = max(0, annual_limit - total_allocated_volume_year)
                        remaining_overall_limit_cost = max(0, annual_limit - total_allocated_cost_year)
                        affordable_vol_overall = remaining_overall_limit_cost / (cost_of_unit + 1e-9)
                        remaining_reduction_target_vol = max(0, target_reduction_alloc - allocated_reduction_value)
                        remaining_reduction_target_cost = max(0, target_reduction_alloc - allocated_reduction_value)
                        affordable_vol_reduction_target = remaining_reduction_target_cost / (cost_of_unit + 1e-9)

                        if constraint_type == "Volume Constrained":
                            vol_to_allocate = min(available_vol, remaining_overall_limit_vol, remaining_reduction_target_vol)
                        else: # Budget Constrained
                            vol_to_allocate = min(available_vol, affordable_vol_overall, affordable_vol_reduction_target)

                        if vol_to_allocate < 1e-6: vol_to_allocate = 0.0
                        if vol_to_allocate > 0:
                            cost = vol_to_allocate * price
                            total_allocated_volume_year += vol_to_allocate
                            total_allocated_cost_year += cost
                            if project_name not in allocated_projects_this_year:
                                allocated_projects_this_year[project_name] = {'volume': 0.0, 'price': price, 'type': 'reduction'}
                            allocated_projects_this_year[project_name]['volume'] += vol_to_allocate
                            year_df.loc[idx, 'volume'] -= vol_to_allocate
                            allocated_reduction_value += (cost if constraint_type == "Budget Constrained" else vol_to_allocate)

 # --- PHASE 2: Gap Filling ---
        # Check if limit was already met in Phase 1
        limit_met_before_phase2 = (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                                   (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6)

        if not limit_met_before_phase2 and annual_limit > 0:
            # Get projects with volume remaining after Phase 1
            # Ensure the 'volume' column in year_df reflects remaining volume accurately before this point.
            # Create a working copy sorted globally (Priority DESC, Price ASC)
            remaining_projects_df = year_df[year_df['volume'] > 1e-6].copy()
            remaining_projects_df.sort_values(by=['priority', 'price'], ascending=[False, True], inplace=True)

            if not remaining_projects_df.empty:
                projects_processed_in_phase2 = set()
                # Store Phase 2 allocations temporarily to add them later
                temp_allocations_phase2 = {} # { proj_name: allocated_vol_phase2 }

                # --- Pass 2.A: Allocate to PRIORITIZED projects sequentially ---
                print(f"    Phase 2A: Processing prioritized projects for gap filling (Year {year})...")
                # Use itertuples for potentially better performance
                for project_tuple in remaining_projects_df.itertuples(index=True): # Get index too
                    idx = project_tuple.Index # Get the actual index label
                    
                    # Check global limit status before processing each project
                    limit_met_phase2a = (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                                       (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6)
                    if limit_met_phase2a:
                        print("      Phase 2A: Annual limit reached.")
                        break # Stop processing prioritized projects

                    # Check if project is prioritized (handle potential NaN/non-numeric)
                    priority_val = getattr(project_tuple, 'priority', None)
                    is_prioritized = pd.notna(priority_val) and isinstance(priority_val, (int, float)) and priority_val > 0

                    if is_prioritized:
                        project_name = getattr(project_tuple, 'project name', 'N/A')
                        # Use the volume directly from the tuple (which comes from the copied df)
                        available_vol = getattr(project_tuple, 'volume', 0.0) 
                        price = getattr(project_tuple, 'price', 0.0)
                        cost_of_unit = price if pd.notna(price) and price > 0 else 0
                        vol_to_allocate = 0.0

                        if available_vol < 1e-6: continue # Skip if no volume left

                        # Calculate allocation based on constraint
                        if constraint_type == "Volume Constrained":
                            remaining_overall_limit_vol = max(0, annual_limit - total_allocated_volume_year)
                            vol_to_allocate = min(available_vol, remaining_overall_limit_vol)
                        else: # Budget Constrained
                            remaining_overall_budget = max(0, annual_limit - total_allocated_cost_year)
                            # Prevent division by zero if cost_of_unit is 0
                            affordable_vol_overall = remaining_overall_budget / (cost_of_unit + 1e-9) if cost_of_unit > 1e-9 else float('inf')
                            vol_to_allocate = min(available_vol, affordable_vol_overall)

                        # Ensure positive allocation and avoid tiny amounts if limits are tight
                        if vol_to_allocate < 1e-6: vol_to_allocate = 0.0

                        if vol_to_allocate > 0:
                            cost = vol_to_allocate * price
                            # Update totals
                            total_allocated_volume_year += vol_to_allocate
                            total_allocated_cost_year += cost
                            # Record allocation for Phase 2
                            if project_name not in temp_allocations_phase2:
                                 temp_allocations_phase2[project_name] = 0.0
                            temp_allocations_phase2[project_name] += vol_to_allocate
                            # Mark as processed in this phase
                            projects_processed_in_phase2.add(project_name)
                            # Decrement remaining volume IN THE WORKING COPY for Pass 2B accuracy
                            remaining_projects_df.loc[idx, 'volume'] -= vol_to_allocate
                    # else: project is non-prioritized, handled in Pass 2B

                # --- Pass 2.B: Allocate to NON-PRIORITIZED projects ---
                print(f"    Phase 2B: Processing non-prioritized projects for gap filling (Year {year})...")
                
                # Filter for non-prioritized projects with remaining volume AFTER Pass 2A
                non_prio_to_fill_df = remaining_projects_df[
                    ~remaining_projects_df['project name'].isin(projects_processed_in_phase2) &
                    (remaining_projects_df['volume'] > 1e-6)
                ].copy() # Create a new copy to work with for non-prio

                if not non_prio_to_fill_df.empty:
                    # Check global limit status again before starting Pass 2B
                    limit_met_before_pass2b = (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                                              (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6)

                    if not limit_met_before_pass2b:

                        # --- BRANCHING LOGIC FOR NON-PRIORITIZED ---

                        # --- Equal Share Logic (Volume Constrained) ---
                        if constraint_type == "Volume Constrained":
                            print("      Applying equal volume share logic (Volume Constrained).")
                            # Get unique types remaining among non-prioritized
                            types_to_fill = non_prio_to_fill_df['project type'].unique()

                            # Process type by type for equal sharing
                            for current_type in types_to_fill:
                                # Check overall limit before processing the type
                                current_limit_vol_start_type = max(0, annual_limit - total_allocated_volume_year)
                                if current_limit_vol_start_type < 1e-6:
                                    print("      Global volume limit met, skipping further types.")
                                    break # Global limit met

                                # Get non-prio projects of this type with remaining volume
                                type_non_prio_mask = (non_prio_to_fill_df['project type'] == current_type) & \
                                                     (non_prio_to_fill_df['volume'] > 1e-6)
                                type_indices = non_prio_to_fill_df.index[type_non_prio_mask]
                                
                                if len(type_indices) == 0: continue # No active projects of this type

                                print(f"        Processing type: {current_type} ({len(type_indices)} projects)")
                                # Use iterative approach for fair sharing with individual caps
                                while True:
                                    current_limit_vol = max(0, annual_limit - total_allocated_volume_year)
                                    if current_limit_vol < 1e-6: break # Check limit at start of each iteration

                                    # Find projects of this type still active (volume > 1e-6)
                                    active_mask_iter = non_prio_to_fill_df.loc[type_indices, 'volume'] > 1e-6
                                    active_indices_iter = type_indices[active_mask_iter]
                                    num_active = len(active_indices_iter)

                                    if num_active == 0: break # Type exhausted or projects too small

                                    # Calculate ideal share based on current remaining limit and active projects
                                    ideal_share_vol = current_limit_vol / num_active

                                    # Find bottleneck: project that will finish first if ideal share is given
                                    # Needed to allocate efficiently in chunks
                                    min_remaining_vol_active = non_prio_to_fill_df.loc[active_indices_iter, 'volume'].min()
                                    
                                    # Volume to allocate per project in this iteration step
                                    vol_per_project_this_iter = min(ideal_share_vol, min_remaining_vol_active)

                                    # Avoid excessively small allocations if limit is almost met
                                    if vol_per_project_this_iter < 1e-9: break

                                    # Ensure we don't allocate more than the remaining limit overall in this step
                                    max_alloc_this_iter = vol_per_project_this_iter * num_active
                                    if max_alloc_this_iter > current_limit_vol + 1e-9: # Add tolerance
                                        # Scale back allocation if calculated total exceeds remaining limit
                                        vol_per_project_this_iter = current_limit_vol / num_active
                                        if vol_per_project_this_iter < 1e-9: break

                                    # Allocate this volume to all active projects of the type
                                    allocated_in_iter = 0
                                    for idx in active_indices_iter:
                                        # Get needed info (safer access)
                                        project_name = non_prio_to_fill_df.loc[idx, 'project name']
                                        price = non_prio_to_fill_df.loc[idx, 'price']
                                        
                                        # Apply allocation
                                        actual_alloc_proj = min(vol_per_project_this_iter, non_prio_to_fill_df.loc[idx, 'volume'])
                                        
                                        if actual_alloc_proj < 1e-9: continue # Skip tiny allocation

                                        non_prio_to_fill_df.loc[idx, 'volume'] -= actual_alloc_proj
                                        
                                        if project_name not in temp_allocations_phase2:
                                            temp_allocations_phase2[project_name] = 0.0
                                        temp_allocations_phase2[project_name] += actual_alloc_proj

                                        # Update global totals
                                        total_allocated_volume_year += actual_alloc_proj
                                        # Track cost even if volume constrained
                                        total_allocated_cost_year += actual_alloc_proj * (price if pd.notna(price) else 0) 
                                        allocated_in_iter += actual_alloc_proj

                                    # Break inner loop if no volume was actually allocated (e.g., due to precision)
                                    if allocated_in_iter < 1e-9:
                                        break


                        # --- Sequential Price Logic (Budget Constrained) ---
                        else: # Budget Constrained
                            print("      Applying sequential price-based logic (Budget Constrained).")
                            # Sort remaining non-prioritized projects by price only (ascending)
                            non_prio_to_fill_df.sort_values(by=['price'], ascending=[True], inplace=True)

                            for idx, project in non_prio_to_fill_df.iterrows():
                                # Check budget limit before processing project
                                if total_allocated_cost_year >= annual_limit - 1e-6:
                                    print("        Budget limit reached.")
                                    break # Stop processing non-prio projects

                                project_name = project['project name']
                                # Use volume remaining after Pass 2A
                                available_vol = project['volume'] 
                                price = project['price']
                                cost_of_unit = price if pd.notna(price) and price > 0 else 0
                                vol_to_allocate = 0.0

                                if available_vol < 1e-6: continue

                                remaining_overall_budget = max(0, annual_limit - total_allocated_cost_year)
                                # Prevent division by zero
                                affordable_vol_overall = remaining_overall_budget / (cost_of_unit + 1e-9) if cost_of_unit > 1e-9 else float('inf')
                                vol_to_allocate = min(available_vol, affordable_vol_overall)

                                if vol_to_allocate < 1e-6: vol_to_allocate = 0.0

                                if vol_to_allocate > 0:
                                    cost = vol_to_allocate * price
                                    # Update totals
                                    total_allocated_volume_year += vol_to_allocate
                                    total_allocated_cost_year += cost
                                    # Record allocation for Phase 2
                                    if project_name not in temp_allocations_phase2:
                                         temp_allocations_phase2[project_name] = 0.0
                                    temp_allocations_phase2[project_name] += vol_to_allocate
                                    # No need to decrement volume in df here as we iterate only once sequentially

                # --- Consolidate Phase 2 Allocations into final result ---
                print(f"    Consolidating Phase 2 allocations for Year {year}...")
                for project_name, allocated_vol_phase2 in temp_allocations_phase2.items():
                     if allocated_vol_phase2 > 1e-6:
                         # Find original project details (price, type) using original year_df
                         # Ensure year_df wasn't modified in ways that lose original info
                         try:
                             # Find the row in the original yearly dataframe
                             original_project_row = year_df[year_df['project name'] == project_name].iloc[0]
                             price = original_project_row['price']
                             ptype = original_project_row['project type'] # Get type from original data

                             # Add or update the main allocation dictionary
                             if project_name not in allocated_projects_this_year:
                                  # If project wasn't allocated in Phase 1, create entry
                                  allocated_projects_this_year[project_name] = {'volume': 0.0, 'price': price, 'type': ptype}
                             
                             # Add Phase 2 volume to any existing Phase 1 volume
                             allocated_projects_this_year[project_name]['volume'] += allocated_vol_phase2
                             # Optional: Round final volume here if needed
                             # allocated_projects_this_year[project_name]['volume'] = round(allocated_projects_this_year[project_name]['volume'], 6) 

                         except IndexError:
                              allocation_warnings.append(f"Warning for {year}: Could not find original details for project '{project_name}' during Phase 2 consolidation.")


        # --- Store final allocation for the year ---
        # The dictionary allocated_projects_this_year now contains combined Phase 1 and Phase 2 allocations.
        portfolio[year] = allocated_projects_this_year
        print(f"  Finished allocation for {year}. Allocated Vol: {total_allocated_volume_year:.2f}, Cost: {total_allocated_cost_year:.2f}")
        # --- End Year Loop --- (This closing bracket was outside the original snippet)


        # --- Display Warnings ---
        if allocation_warnings:
            st.warning("Allocation Notes & Warnings:")
            for warning in allocation_warnings:
                st.markdown(f"- {warning}")

        # --- Portfolio Analysis and Visualization ---
        all_types_in_portfolio = set()
        portfolio_data_list = []
        for year, projects in portfolio.items():
             if projects: # Only process years where allocation happened
                for name, info in projects.items():
                    all_types_in_portfolio.add(info['type'])
                    portfolio_data_list.append({
                        'year': year,
                        'project name': name,
                        'type': info['type'],
                        'volume': info['volume'],
                        'price': info['price'],
                        'cost': info['volume'] * info['price']
                    })

        if not portfolio_data_list:
             st.error("No projects could be allocated based on the selected criteria and available data.")
             st.stop()

        portfolio_df = pd.DataFrame(portfolio_data_list)

        # Aggregations for plotting and summary
        summary_list = []
        for year in selected_years:
            year_data = portfolio_df[portfolio_df['year'] == year]
            total_volume = year_data['volume'].sum()
            total_cost = year_data['cost'].sum()
            avg_price = total_cost / (total_volume + 1e-9)
            volume_by_type = year_data.groupby('type')['volume'].sum()
            summary_entry = {'Year': year, 'Total Volume (tonnes)': total_volume, 'Total Cost (€)': total_cost, 'Average Price (€/tonne)': avg_price, 'Target Constraint': annual_constraints.get(year, 0)}
            # Add volumes for all potentially selected types, defaulting to 0 if not in this year's allocation
            for proj_type in (global_removal_types + ([global_reduction_type] if global_reduction_type else [])):
                  summary_entry[f'Volume {proj_type.capitalize()}'] = volume_by_type.get(proj_type, 0)
            summary_list.append(summary_entry)
        summary_df = pd.DataFrame(summary_list)

        st.subheader("Portfolio Composition & Price Over Time")
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Define UPDATED consistent colors
        color_map = {
            'technical removal': '#64B5F6', # Blueish (Material Blue Lighten 2)
            'natural removal': '#81C784', # Greenish (Material Green Lighten 2)
            'reduction': '#B0BEC5', # Greyish (Material Blue Grey Lighten 1)
            # Add more types and colors if needed
        }
        default_color = '#D3D3D3' # Light Grey for unknown types

        # Plot types present in the actual allocation
        plot_types = sorted([t for t in all_types_in_portfolio if t in color_map])

        for type_name in plot_types:
            type_volume_col = f'Volume {type_name.capitalize()}'
            if type_volume_col in summary_df.columns:
                type_volume = summary_df[type_volume_col]
                fig.add_trace(go.Bar(x=summary_df['Year'], y=type_volume, name=type_name.capitalize(), marker_color=color_map.get(type_name, default_color)), secondary_y=False)

        fig.add_trace(go.Scatter(x=summary_df['Year'], y=summary_df['Average Price (€/tonne)'], name='Avg Price (€/tonne)', marker=dict(symbol='circle'), line=dict(color='#546E7A')), secondary_y=True) # Darker Blue Grey for price line

        fig.update_layout(xaxis_title='Year', yaxis_title='Volume (tonnes)', yaxis2_title='Average Price (€/tonne)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), barmode='stack', template="plotly_white", yaxis=dict(rangemode='tozero'), yaxis2=dict(rangemode='tozero'))
        st.plotly_chart(fig, use_container_width=True)

        # Final Year Removal %
        if end_year in summary_df['Year'].values:
             final_year_summary = summary_df[summary_df['Year'] == end_year].iloc[0]
             final_tech = final_year_summary.get(f'Volume Technical removal', 0) if 'Volume Technical removal' in final_year_summary else 0
             final_nat = final_year_summary.get(f'Volume Natural removal', 0) if 'Volume Natural removal' in final_year_summary else 0
             final_total_removal = final_tech + final_nat
             final_total = final_year_summary['Total Volume (tonnes)']
             achieved_removal_perc = (final_total_removal / (final_total + 1e-9)) * 100
             if global_has_removals: st.metric(label=f"Achieved Removal % in {end_year}", value=f"{achieved_removal_perc:.2f}%")
             elif not selected_projects: st.info("No projects selected.")
             else: st.info("No removal projects selected.")
        else:
             st.info(f"No data allocated for the final year ({end_year}).")

        st.subheader("Yearly Summary")
        summary_display_df = summary_df.copy()
        # Standardize column names before display
        summary_display_df['Target'] = summary_display_df['Target Constraint']
        summary_display_df['Achieved Volume'] = summary_display_df['Total Volume (tonnes)']
        summary_display_df['Achieved Cost (€)'] = summary_display_df['Total Cost (€)']
        summary_display_df['Avg Price (€/tonne)'] = summary_display_df['Average Price (€/tonne)']

        if constraint_type == "Volume Constrained":
            summary_display_df.rename(columns={'Target': 'Target Volume'}, inplace=True)
            display_cols = ['Year', 'Target Volume', 'Achieved Volume', 'Achieved Cost (€)', 'Avg Price (€/tonne)']
        else:
             summary_display_df.rename(columns={'Target': 'Target Budget (€)'}, inplace=True)
             display_cols = ['Year', 'Target Budget (€)', 'Achieved Cost (€)', 'Achieved Volume', 'Avg Price (€/tonne)']

        # Add type columns to display if they exist in the summary
        for proj_type in plot_types:
             col_name = f'Volume {proj_type.capitalize()}'
             if col_name in summary_display_df.columns:
                  display_cols.append(col_name)
                  summary_display_df[col_name] = summary_display_df[col_name].map('{:,.0f}'.format) # Format volume

        # Format numeric columns
        for col in ['Target Budget (€)', 'Achieved Cost (€)', 'Avg Price (€/tonne)']:
             if col in summary_display_df.columns: summary_display_df[col] = summary_display_df[col].map('{:,.2f}'.format)
        for col in ['Target Volume', 'Achieved Volume']:
              if col in summary_display_df.columns: summary_display_df[col] = summary_display_df[col].map('{:,.0f}'.format)


        st.dataframe(summary_display_df[display_cols].set_index('Year'))


        # --- Convert Portfolio Dictionary to DataFrame (Summing volumes across all years) ---
        records = []
        for year, projects in portfolio.items():
            for name, info in projects.items():
                if info['volume'] > 0:
                    records.append({
                        'year': str(year),
                        'type': info['type'],
                        'project': name,
                        'volume': info['volume']
                    })
        
        df = pd.DataFrame(records)
        
        # --- Aggregate the Data (sum volumes across all years for each project) ---
        df_summed = df.groupby(['type', 'project']).agg({'volume': 'sum'}).reset_index()
        
        # --- Sunburst Plot for Total Allocation Across All Years ---
        fig = px.sunburst(
            df_summed,
            path=['type', 'project'],  # Hierarchical structure: Type -> Project
            values='volume',           # Size based on the total volume across all years
            color='type',              # Color by type (Technical, Natural, Reduction)
            color_discrete_map={
                'technical removal': 'royalblue',
                'natural removal': 'forestgreen',
                'reduction': 'orange'
            },
            title="Total Carbon Credit Allocation Across All Years"
        )
        
        fig.update_traces(textinfo='label+value+percent entry')
        
        # --- Display the Chart in Streamlit ---
        st.subheader("📊 Total Allocation Overview (Across All Years Combined)")
        st.plotly_chart(fig, use_container_width=True)

    
        # Raw Allocation Data
        st.subheader("Detailed Allocation Data")
        if st.checkbox("Show raw project allocations by year", key="show_raw"):
             display_portfolio_df = portfolio_df.copy()
             display_portfolio_df['volume'] = display_portfolio_df['volume'].map('{:,.2f}'.format)
             display_portfolio_df['price'] = display_portfolio_df['price'].map('{:,.2f}'.format)
             display_portfolio_df['cost'] = display_portfolio_df['cost'].map('{:,.2f}'.format)
             st.dataframe(display_portfolio_df[['year', 'project name', 'type', 'volume', 'price', 'cost']].sort_values(by=['year', 'project name']))

    elif df_upload:
         st.warning("Please select at least one project in Step 2 to build the portfolio.")
