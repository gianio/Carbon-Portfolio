import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- Streamlit App Configuration ---
# Use layout='wide' for more space
st.set_page_config(layout="wide")

# Inject CSS for custom styling (Fonts, Colors, Inputs, Spacing)
st.markdown("""
    <style>
        /* --- General Font and Background --- */
        body, .stApp {
            background-color: #F0F2F6; /* Light grey background */
            font-family: 'Calibri', sans-serif; /* Set default font */
        }
        /* Increase base font size for the app */
        body, .stApp, .stApp p, .stApp label, .stApp input, .stApp select, .stApp textarea, .stApp button, .stApp div[data-baseweb="select"] > div, .stDataFrame div {
            font-size: 1.08rem !important; /* Slightly increased font size */
            font-family: 'Calibri', sans-serif !important; /* Ensure font */
        }
        /* Ensure Radio buttons also use the font */
        .stRadio label {
             font-size: 1.08rem !important;
             font-family: 'Calibri', sans-serif !important;
        }

        /* --- Input Field Styling --- */
        /* Target various input types */
        .stTextInput input,
        .stNumberInput input,
        .stDateInput input,
        .stTimeInput input,
        .stSelectbox select,
        .stMultiselect div[data-baseweb="input"] { /* Target the div containing the multiselect input area */
            background-color: #FFFFFF !important; /* White background */
            color: #262730 !important; /* Darker text for readability */
            border: 1px solid #DCDCDC !important; /* Subtle border */
            border-radius: 0.3rem !important; /* Slightly more rounded */
            font-family: 'Calibri', sans-serif !important; /* Ensure font */
            font-size: 1.08rem !important; /* Ensure font size */
        }
        /* Ensure number input label font */
        .stNumberInput label {
             font-family: 'Calibri', sans-serif !important;
             font-size: 1.08rem !important;
             color: #31333F; /* Consistent label color */
        }
         /* Adjust label padding/margin if needed */
         label[data-baseweb="form-control-label"] {
              margin-bottom: 4px !important; /* Slightly less space below labels */
              display: block; /* Ensure label takes full width */
         }


        /* --- Slider Styling --- */
        div[data-baseweb="slider"] {
            /* Color of the thumb (circle) */
            & > div:nth-last-child(2) { /* Selector for the thumb */
                 background-color: #A2C244 !important; /* Custom green/yellow */
                 border: 2px solid #FFFFFF !important; /* Add white border to thumb */
                 box-shadow: 0 1px 3px rgba(0,0,0,0.2); /* Add subtle shadow */
            }
            /* Color of the track's filled part */
            & > div:nth-child(2) > div { /* Selector for the filled track */
                background-color: #A2C244 !important;
            }
            /* Color of the track's unfilled part */
            & > div:nth-child(1) {
                 background-color: #DCDCDC !important; /* Light grey for unfilled track */
            }
        }

        /* --- Multiselect Styling --- */
        /* Background of selected items (pills) */
        .stMultiSelect span[data-baseweb="tag"] {
            background-color: #A2C244 !important; /* Custom green/yellow */
            color: #FFFFFF !important; /* White text on the pill */
            font-family: 'Calibri', sans-serif !important; /* Ensure font */
            font-size: 1.0rem !important; /* Slightly smaller font for pills */
            border-radius: 0.5rem !important; /* Make pills rounder */
            padding-top: 2px !important; /* Adjust padding */
            padding-bottom: 2px !important;
            margin-right: 5px !important; /* Space between pills */
        }
         /* Change border color when multiselect is focused */
        .stMultiSelect div[data-baseweb="input"]:focus-within { /* Target wrapper on focus */
           border: 1px solid #A2C244 !important; /* Highlight border */
           box-shadow: 0 0 0 2px rgba(162, 194, 68, 0.2) !important; /* Focus ring */
        }
        /* Ensure font in dropdown list */
         div[data-baseweb="popover"] ul[role="listbox"] li {
            font-family: 'Calibri', sans-serif !important;
            font-size: 1.08rem !important;
         }


        /* --- Headers and Titles --- */
        h1 { /* Main Title */
            font-weight: bold;
            font-family: 'Calibri', sans-serif !important;
            font-size: 2.3rem !important; /* Adjusted title size */
            color: #31333F;
            text-align: center; /* Center title */
            margin-bottom: 25px; /* Space below title */
        }
        h2 { /* Subheaders */
            border-bottom: 2px solid #A2C244; /* Custom green/yellow underline */
            padding-bottom: 8px; /* More space below text */
            margin-top: 35px; /* More space above */
            margin-bottom: 20px; /* More space below */
            font-weight: bold;
            font-family: 'Calibri', sans-serif !important;
            font-size: 1.6rem !important; /* Larger subheaders */
             color: #31333F;
        }
         h3 { /* Sub-subheaders (e.g., Steps) */
            font-weight: bold;
            font-family: 'Calibri', sans-serif !important;
            font-size: 1.3rem !important;
             color: #31333F;
             margin-top: 15px;
             margin-bottom: 10px;
         }
         /* Markdown text styling */
         .stMarkdown p, .stMarkdown li {
            font-size: 1.08rem !important;
            font-family: 'Calibri', sans-serif !important;
         }


        /* --- General Spacing & Layout --- */
        .stApp > header { display: none; } /* Hide default Streamlit header */
        .stButton>button { margin-top: 15px; font-size: 1.08rem !important; font-family: 'Calibri', sans-serif !important;} /* Add margin above buttons & style */
        /* Reduce vertical space taken by number inputs in columns */
         div.stColumn > div > div > div[data-testid="element-container"] {
              padding-top: 0.3rem !important;
              padding-bottom: 0.3rem !important;
         }
         /* Center align radio buttons */
         .stRadio > div {
            flex-direction: row !important;
            justify-content: center !important;
            gap: 20px; /* Add space between radio buttons */
         }


    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.title("🌱 Multi-Year Carbon Portfolio Builder")

# --- Data Input ---
df_upload = st.file_uploader("Upload project CSV", type="csv")

# --- Main Logic ---
if df_upload:
    # --- Data Reading and Initial Validation ---
    try:
        data = pd.read_csv(df_upload)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    required_cols = ['project name', 'project type']
    if not all(col in data.columns for col in required_cols):
        st.error(f"CSV missing essential columns: {', '.join(required_cols)}")
        st.stop()

    # --- Project Overview Section ---
    st.subheader("Project Overview")
    with st.container(): # Use container for better layout control if needed
        years = st.number_input("Portfolio Duration (Years):", min_value=1, max_value=20, value=6, step=1, help="Number of years starting from 2025")
        start_year = 2025
        end_year = start_year + years - 1
        selected_years = list(range(start_year, end_year + 1))

        overview = data.copy()
        price_cols = [f"price {year}" for year in selected_years if f"price {year}" in overview.columns]

        # Ensure 'priority' exists and handle NaNs
        if 'priority' not in overview.columns: overview['priority'] = 50
        else: overview['priority'] = overview['priority'].fillna(50)

        # Calculate average price safely
        if price_cols:
            for col in price_cols:
                overview[col] = pd.to_numeric(overview[col], errors='coerce')
            # Calculate mean only if there are non-NaN prices
            overview['avg_price'] = overview[price_cols].mean(axis=1, skipna=True)
            overview_display_cols = ['project name', 'project type']
            if 'description' in overview.columns: overview_display_cols.append('description')
            overview_display_cols.append('avg_price')
            # Format avg_price for display
            overview_display = overview[overview_display_cols].drop_duplicates(subset=['project name']).reset_index(drop=True)
            if 'avg_price' in overview_display.columns:
                 overview_display['avg_price'] = overview_display['avg_price'].map('{:,.2f}'.format) # Format after drop_duplicates
            st.dataframe(overview_display)

        else:
            overview_display_cols = ['project name', 'project type']
            if 'description' in overview.columns: overview_display_cols.append('description')
            st.dataframe(overview[overview_display_cols].drop_duplicates(subset=['project name']).reset_index(drop=True))
            st.caption(f"Note: No 'price YEAR' columns found for {start_year}-{end_year}. Average price not calculated.")

    # --- Portfolio Settings Section ---
    st.subheader("Step 1: Define Portfolio Settings")
    constraint_type = st.radio("Constraint type:", ["Volume Constrained", "Budget Constrained"], key="constraint_type", horizontal=True)
    annual_constraints = {}

    # Use columns for layout
    col_settings_1, col_settings_2 = st.columns([0.6, 0.4]) # Input constraints | Strategy Sliders

    with col_settings_1:
        st.markdown("##### Annual Constraints")
        # Max 6 columns shown at once for yearly inputs
        cols_per_year = st.columns(min(years, 6))
        col_idx = 0
        if constraint_type == "Volume Constrained":
            # Removed descriptive text "Enter annual purchase volumes (tonnes):"
            for year in selected_years:
                with cols_per_year[col_idx % len(cols_per_year)]:
                    default_val = 1000
                    annual_constraints[year] = st.number_input(
                        f"{year}", min_value=0, step=100, value=default_val, key=f"vol_{year}",
                        help="Target volume in tonnes for this year"
                    )
                col_idx += 1
        else: # Budget Constrained
            # Removed descriptive text "Enter annual budget (€):"
            for year in selected_years:
                 with cols_per_year[col_idx % len(cols_per_year)]:
                    default_val = 10000
                    annual_constraints[year] = st.number_input(
                        f"{year} (€)", min_value=0, step=1000, value=default_val, key=f"bud_{year}",
                        help="Target budget in Euro for this year"
                    )
                 col_idx += 1

    with col_settings_2:
        st.markdown("##### Portfolio Strategy")
        removal_target = st.slider(f"Target Removal % by {end_year}", 0, 100, 80, key="removal_target") / 100
        transition_speed = st.slider("Transition Speed (1=Slow, 10=Fast)", 1, 10, 5, key="transition_speed")
        removal_preference = st.slider("Preference (1=Natural, 10=Technical)", 1, 10, 5, key="removal_preference", help="Influences allocation order & Phase 1 targets in mixed portfolios")

    # --- Project Selection Section ---
    st.subheader("Step 2: Select and Prioritize Projects")
    project_names = sorted(data['project name'].unique().tolist())
    selected_projects = st.multiselect("Select projects to include:", project_names, default=project_names, key="select_proj")
    favorite_projects = st.multiselect("Select favorite projects (+10 priority boost):", selected_projects, key="fav_proj") # Simplified boost description

    # --- Allocation Section ---
    if selected_projects:
        selected_df = data[data['project name'].isin(selected_projects)].copy()

        # Apply favorite boost safely
        if favorite_projects and 'priority' in selected_df.columns:
            selected_df['priority'] = selected_df.apply(
                lambda row: row['priority'] + 10 if row['project name'] in favorite_projects else row['priority'],
                axis=1
            )
            selected_df['priority'] = selected_df['priority'].clip(upper=100)

        # Identify globally available types from selection
        global_selected_types = selected_df['project type'].unique().tolist()
        global_removal_types = [t for t in global_selected_types if t in ['technical removal', 'natural removal']]
        global_reduction_type = 'reduction' if 'reduction' in global_selected_types else None
        global_has_removals = bool(global_removal_types)
        global_has_reductions = bool(global_reduction_type)

        # --- Allocation Logic ---
        portfolio = {year: {} for year in selected_years}
        allocation_warnings = []

        # --- Start Year Loop ---
        for year_idx, year in enumerate(selected_years):
            year_str = str(year)
            volume_col = f"available volume {year_str}"
            price_col = f"price {year_str}"

            # Check required columns for the year
            if volume_col not in selected_df.columns or price_col not in selected_df.columns:
                allocation_warnings.append(f"Warning {year}: Missing '{volume_col}' or '{price_col}'. Skipping.")
                portfolio[year] = {}
                continue

            # Prepare data for the current year
            year_df = selected_df[['project name', 'project type', 'priority', volume_col, price_col]].copy()
            year_df.rename(columns={volume_col: 'volume', price_col: 'price'}, inplace=True)
            year_df['volume'] = pd.to_numeric(year_df['volume'], errors='coerce').fillna(0)
            year_df['price'] = pd.to_numeric(year_df['price'], errors='coerce')
            year_df = year_df[(year_df['volume'] > 0) & pd.notna(year_df['price'])] # Filter valid projects

            if year_df.empty: # Skip if no valid projects this year
                portfolio[year] = {}; continue

            # Get annual constraint for this year
            annual_limit = annual_constraints.get(year, 0)
            if annual_limit <= 0: # Skip if constraint is zero
                 portfolio[year] = {}; continue

            # Initialize year variables
            total_allocated_volume_year = 0.0
            total_allocated_cost_year = 0.0
            allocated_projects_this_year = {}

            # Calculate dynamic removal target fraction
            year_fraction = (year_idx + 1) / years
            current_removal_target_fraction = removal_target * (year_fraction ** (0.5 + 0.1 * transition_speed))

            # Determine portfolio type for this year based on *available* projects
            current_year_types = year_df['project type'].unique().tolist()
            current_has_removals = any(t in global_removal_types for t in current_year_types)
            current_has_reductions = global_reduction_type in current_year_types
            current_only_removals = current_has_removals and not current_has_reductions
            current_only_reductions = not current_has_removals and current_has_reductions

            # Adjust target fraction based on available types
            if current_only_reductions: current_removal_target_fraction = 0.0
            elif current_only_removals: current_removal_target_fraction = 1.0
            elif not current_has_removals and not current_has_reductions:
                 portfolio[year] = {}; continue # No valid types available

            # Calculate target budget/volume for removals vs reductions for Phase 1 guidance
            target_removal_alloc = annual_limit * current_removal_target_fraction
            target_reduction_alloc = annual_limit * (1.0 - current_removal_target_fraction)

            # Sort available projects by priority (desc) and price (asc)
            year_df.sort_values(by=['priority', 'price'], ascending=[False, True], inplace=True)
            removals_df = year_df[year_df['project type'].isin(global_removal_types)].copy() if current_has_removals else pd.DataFrame()
            reductions_df = year_df[year_df['project type'] == global_reduction_type].copy() if current_has_reductions else pd.DataFrame()

            # --- PHASE 1: Targeted Allocation ---
            # Tries to meet targets set by sliders, respecting preferences, within the overall annual limit.
            allocated_natural_value = 0.0 # Track value (vol or cost) allocated per type in this phase
            allocated_technical_value = 0.0
            allocated_reduction_value = 0.0

            # 1.a Allocate Removals
            if current_has_removals and not removals_df.empty:
                removal_pref_factor = removal_preference / 10.0
                natural_pref_target = target_removal_alloc * (1.0 - removal_pref_factor) # Ideal target for this type
                technical_pref_target = target_removal_alloc * removal_pref_factor # Ideal target for this type
                removal_order_types = ['natural removal', 'technical removal']
                if removal_preference > 5: removal_order_types.reverse() # Determine allocation order

                # Loop through preferred removal type first, then the other
                for r_type in removal_order_types:
                    if r_type not in removals_df['project type'].unique(): continue
                    type_df = removals_df[removals_df['project type'] == r_type]

                    # Loop through projects of the current type
                    for idx, project in type_df.iterrows():
                        # Check if overall annual limit is already met
                        if (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                           (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6): break

                        # Get project details and remaining volume
                        project_name = project['project name']
                        available_vol = year_df.loc[idx, 'volume'] # Use .loc to get current state
                        price = project['price']
                        if available_vol < 1e-6: continue # Skip if already used up

                        # Calculate volume to allocate based on constraints
                        vol_to_allocate = 0.0
                        remaining_overall_limit_vol = max(0, annual_limit - total_allocated_volume_year)
                        remaining_overall_limit_cost = max(0, annual_limit - total_allocated_cost_year)
                        cost_of_unit = price if price > 0 else 0
                        affordable_vol_overall = remaining_overall_limit_cost / (cost_of_unit + 1e-9)

                        if constraint_type == "Volume Constrained":
                            if current_only_removals: # Focus only on overall limit
                                vol_to_allocate = min(available_vol, remaining_overall_limit_vol)
                            else: # Mixed portfolio: Limit by overall limit AND type-specific target for Phase 1
                                remaining_pref_target_vol = max(0, (natural_pref_target if r_type == 'natural removal' else technical_pref_target) - \
                                                                   (allocated_natural_value if r_type == 'natural removal' else allocated_technical_value))
                                vol_to_allocate = min(available_vol, remaining_overall_limit_vol, remaining_pref_target_vol)
                        else: # Budget Constrained
                            if current_only_removals: # Focus only on overall limit
                                vol_to_allocate = min(available_vol, affordable_vol_overall)
                            else: # Mixed portfolio: Limit by overall limit AND type-specific target budget
                                remaining_pref_target_cost = max(0, (natural_pref_target if r_type == 'natural removal' else technical_pref_target) - \
                                                                    (allocated_natural_value if r_type == 'natural removal' else allocated_technical_value))
                                affordable_vol_pref = remaining_pref_target_cost / (cost_of_unit + 1e-9)
                                vol_to_allocate = min(available_vol, affordable_vol_overall, affordable_vol_pref)

                        # Final check and allocation
                        if vol_to_allocate < 1e-6 : vol_to_allocate = 0.0
                        if vol_to_allocate > 0:
                            cost = vol_to_allocate * price
                            # Update overall totals
                            total_allocated_volume_year += vol_to_allocate
                            total_allocated_cost_year += cost
                            # Update value allocated for this specific removal type in Phase 1
                            phase1_value = vol_to_allocate if constraint_type == "Volume Constrained" else cost
                            if r_type == 'natural removal': allocated_natural_value += phase1_value
                            else: allocated_technical_value += phase1_value
                            # Record allocation
                            if project_name not in allocated_projects_this_year: allocated_projects_this_year[project_name] = {'volume': 0.0, 'price': price, 'type': r_type}
                            allocated_projects_this_year[project_name]['volume'] += vol_to_allocate
                            # Decrement available volume in the main dataframe for Phase 2
                            year_df.loc[idx, 'volume'] -= vol_to_allocate

                    # Check limits again after processing all projects of one removal type
                    if (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                       (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6): break


            # 1.b Allocate Reductions
            if current_has_reductions and not reductions_df.empty:
                 # Proceed only if the overall limit wasn't already met by removals
                 if not ((constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                         (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6)):
                    # Loop through reduction projects
                    for idx, project in reductions_df.iterrows():
                        # Check limits before processing project
                        if (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                           (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6): break

                        project_name = project['project name']; available_vol = year_df.loc[idx, 'volume']; price = project['price']
                        if available_vol < 1e-6: continue

                        # Calculate volume to allocate based on limits
                        vol_to_allocate = 0.0; cost_of_unit = price if price > 0 else 0
                        remaining_overall_limit_vol = max(0, annual_limit - total_allocated_volume_year)
                        remaining_overall_limit_cost = max(0, annual_limit - total_allocated_cost_year)
                        affordable_vol_overall = remaining_overall_limit_cost / (cost_of_unit + 1e-9)
                        # Phase 1 Reductions also limited by the reduction target
                        remaining_reduction_target_vol = max(0, target_reduction_alloc - allocated_reduction_value)
                        remaining_reduction_target_cost = max(0, target_reduction_alloc - allocated_reduction_value) # Assuming target_reduction_alloc is cost if Budget Constrained
                        affordable_vol_reduction_target = remaining_reduction_target_cost / (cost_of_unit + 1e-9)

                        if constraint_type == "Volume Constrained": vol_to_allocate = min(available_vol, remaining_overall_limit_vol, remaining_reduction_target_vol)
                        else: vol_to_allocate = min(available_vol, affordable_vol_overall, affordable_vol_reduction_target)

                        # Final check and allocation
                        if vol_to_allocate < 1e-6 : vol_to_allocate = 0.0
                        if vol_to_allocate > 0:
                            cost = vol_to_allocate * price
                            # Update totals
                            total_allocated_volume_year += vol_to_allocate; total_allocated_cost_year += cost
                            # Update value allocated to reductions in Phase 1
                            allocated_reduction_value += vol_to_allocate if constraint_type == "Volume Constrained" else cost
                            # Record allocation
                            if project_name not in allocated_projects_this_year: allocated_projects_this_year[project_name] = {'volume': 0.0, 'price': price, 'type': global_reduction_type}
                            allocated_projects_this_year[project_name]['volume'] += vol_to_allocate
                            # Decrement available volume
                            year_df.loc[idx, 'volume'] -= vol_to_allocate

            # --- PHASE 2: Gap Filling ---
            # If the annual limit wasn't met in Phase 1, try to fill using any remaining project volume,
            # sorted purely by priority/price, ignoring original type targets.
            limit_met = (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                        (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6)

            if not limit_met and annual_limit > 0:
                # Get projects with remaining volume and re-sort by priority/price
                remaining_projects_df = year_df[year_df['volume'] > 1e-6].sort_values(by=['priority', 'price'], ascending=[False, True])
                if not remaining_projects_df.empty:
                    # Loop through best remaining projects
                    for idx, project in remaining_projects_df.iterrows():
                        # Check if limit met before processing
                        if (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                           (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6): break

                        project_name = project['project name']; available_vol = project['volume']; price = project['price'] # Get current remaining vol
                        vol_to_allocate = 0.0; cost_of_unit = price if price > 0 else 0

                        # Calculate allocation based only on remaining overall limit
                        if constraint_type == "Volume Constrained":
                            remaining_overall_limit = max(0, annual_limit - total_allocated_volume_year)
                            vol_to_allocate = min(available_vol, remaining_overall_limit)
                        else: # Budget Constrained
                            remaining_overall_budget = max(0, annual_limit - total_allocated_cost_year)
                            affordable_vol_overall = remaining_overall_budget / (cost_of_unit + 1e-9)
                            vol_to_allocate = min(available_vol, affordable_vol_overall)

                        # Final check and allocation
                        if vol_to_allocate < 1e-6 : vol_to_allocate = 0.0
                        if vol_to_allocate > 0:
                            cost = vol_to_allocate * price
                            # Update totals
                            total_allocated_volume_year += vol_to_allocate; total_allocated_cost_year += cost
                            # Add/update allocation record (handles projects chosen in Phase 1 and topped up here)
                            if project_name not in allocated_projects_this_year: allocated_projects_this_year[project_name] = {'volume': 0.0, 'price': price, 'type': project['project type']}
                            allocated_projects_this_year[project_name]['volume'] += vol_to_allocate
                            # Optional: Decrement year_df volume if tracking needed after Phase 2
                            # year_df.loc[idx, 'volume'] -= vol_to_allocate

            # --- Store Results and Handle Warnings for the Year ---
            portfolio[year] = allocated_projects_this_year
            # Check if limit was met (within tolerance) or if target was zero
            final_limit_met = (annual_limit <=0) or \
                              (constraint_type == "Volume Constrained" and abs(total_allocated_volume_year - annual_limit) < 1e-6) or \
                              (constraint_type == "Budget Constrained" and abs(total_allocated_cost_year - annual_limit) < 1e-6)

            # Add warning only if target > 0 and limit not met
            if not final_limit_met and annual_limit > 0:
                limit_partially_met = (total_allocated_volume_year > 1e-6) if constraint_type == "Volume Constrained" else (total_allocated_cost_year > 1e-6)
                if not limit_partially_met: # No allocation happened at all
                    allocation_warnings.append(f"Warning {year}: Could not allocate any volume/budget. Target: {annual_limit:,.2f}. Check project availability/prices.")
                else: # Partial allocation
                     if constraint_type == "Volume Constrained": allocation_warnings.append(f"Warning {year}: Allocated {total_allocated_volume_year:,.0f} / {annual_limit:,.0f} tonnes (insufficient supply).")
                     else: allocation_warnings.append(f"Warning {year}: Spent €{total_allocated_cost_year:,.2f} / €{annual_limit:,.2f} budget (insufficient supply).")
        # --- End of Year Loop ---

        # --- Display Allocation Warnings ---
        if allocation_warnings:
            st.warning("Allocation Notes & Warnings:")
            # Use columns for warnings if there are many? Or just list them.
            for warning in allocation_warnings:
                st.markdown(f"- {warning}")

        # --- Portfolio Analysis and Visualization ---
        # Aggregate results from the 'portfolio' dictionary
        all_types_in_portfolio = set()
        portfolio_data_list = []
        for year, projects in portfolio.items():
             if projects: # Only include years where allocation happened
                for name, info in projects.items():
                    # Ensure volume is positive before adding
                    if info.get('volume', 0) > 1e-6:
                        all_types_in_portfolio.add(info['type'])
                        portfolio_data_list.append({
                            'year': year, 'project name': name, 'type': info['type'],
                            'volume': info['volume'], 'price': info['price'],
                            'cost': info['volume'] * info.get('price', 0) # Handle potential missing price? (Shouldn't happen)
                        })

        # Check if any allocation occurred across all years
        if not portfolio_data_list:
             st.error("No projects could be allocated across the selected years based on criteria and data.")
             st.stop() # Stop if nothing was allocated

        portfolio_df = pd.DataFrame(portfolio_data_list)

        # Create summary DataFrame for plots and tables
        summary_list = []
        plot_types_present = sorted(list(all_types_in_portfolio)) # Types actually allocated

        for year in selected_years:
            year_data = portfolio_df[portfolio_df['year'] == year]
            total_volume = year_data['volume'].sum()
            total_cost = year_data['cost'].sum()
            avg_price = total_cost / (total_volume + 1e-9) # Avoid division by zero
            volume_by_type = year_data.groupby('type')['volume'].sum()
            summary_entry = {
                'Year': year, 'Total Volume (tonnes)': total_volume, 'Total Cost (€)': total_cost,
                'Average Price (€/tonne)': avg_price, 'Target Constraint': annual_constraints.get(year, 0)
            }
            # Add volumes for allocated types, default to 0
            for proj_type in plot_types_present:
                  summary_entry[f'Volume {proj_type.capitalize()}'] = volume_by_type.get(proj_type, 0)
            summary_list.append(summary_entry)
        summary_df = pd.DataFrame(summary_list)

        # --- Plotting Section ---
        st.subheader("Portfolio Composition & Price Over Time")
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Updated Color Map
        color_map = {
            'technical removal': '#64B5F6', # Blueish
            'natural removal': '#81C784', # Greenish
            'reduction': '#B0BEC5', # Greyish
        }
        default_color = '#D3D3D3' # Fallback color

        # Plot bars for each type present in the allocation
        for type_name in plot_types_present:
            type_volume_col = f'Volume {type_name.capitalize()}'
            if type_volume_col in summary_df.columns:
                type_volume = summary_df[type_volume_col]
                fig.add_trace(go.Bar(
                    x=summary_df['Year'], y=type_volume, name=type_name.capitalize(),
                    marker_color=color_map.get(type_name, default_color),
                    hovertemplate = f"<b>Year:</b> %{{x}}<br><b>Type:</b> {type_name.capitalize()}<br><b>Volume:</b> %{{y:,.0f}} tonnes<extra></extra>" # Custom hover text
                    ), secondary_y=False)

        # Plot average price line
        fig.add_trace(go.Scatter(
            x=summary_df['Year'], y=summary_df['Average Price (€/tonne)'], name='Avg Price (€/tonne)',
            marker=dict(symbol='circle', size=8), line=dict(color='#546E7A', width=2.5), # Darker Blue Grey price line
            mode='lines+markers',
            hovertemplate = f"<b>Year:</b> %{{x}}<br><b>Avg Price:</b> €%{{y:,.2f}}/tonne<extra></extra>"
            ), secondary_y=True)

        # Configure layout
        fig.update_layout(
            xaxis_title='Year', yaxis_title='Volume (tonnes)', yaxis2_title='Average Price (€/tonne)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Calibri, sans-serif", size=14)), # Style legend
            barmode='stack', template="plotly_white",
            yaxis=dict(rangemode='tozero', titlefont=dict(family="Calibri, sans-serif", size=16), tickfont=dict(family="Calibri, sans-serif", size=14)), # Style Y1 axis
            yaxis2=dict(rangemode='tozero', titlefont=dict(family="Calibri, sans-serif", size=16), tickfont=dict(family="Calibri, sans-serif", size=14)), # Style Y2 axis
            xaxis=dict(titlefont=dict(family="Calibri, sans-serif", size=16), tickfont=dict(family="Calibri, sans-serif", size=14)), # Style X axis
            hovermode="x unified", # Show hover info for all traces at once
            margin=dict(l=60, r=40, t=50, b=40) # Adjust plot margins
            )
        st.plotly_chart(fig, use_container_width=True) # Ensure chart uses available width

        # --- Final Year Metrics ---
        # Check if final year exists in summary (might not if no allocation)
        if end_year in summary_df['Year'].values:
             final_year_summary = summary_df[summary_df['Year'] == end_year].iloc[0]
             # Safely get volumes, defaulting to 0 if column doesn't exist
             final_tech = final_year_summary.get(f'Volume Technical removal', 0)
             final_nat = final_year_summary.get(f'Volume Natural removal', 0)
             final_total_removal = final_tech + final_nat
             final_total = final_year_summary['Total Volume (tonnes)']
             # Calculate percentage safely
             achieved_removal_perc = (final_total_removal / (final_total + 1e-9)) * 100

             # Display metric only if removals were part of the selected projects
             if global_has_removals:
                 st.metric(label=f"Achieved Removal % in {end_year}", value=f"{achieved_removal_perc:.1f}%")
             elif not selected_projects: st.info("Info: No projects were selected for the portfolio.") # Should not happen if we have results
             else: st.info("Info: No removal type projects were selected globally.")
        else:
             st.info(f"Info: No allocation data available for the final year ({end_year}).")


        # --- Summary Table Section ---
        st.subheader("Yearly Summary")
        summary_display_df = summary_df.copy()
        # Standardize column names before display & formatting
        summary_display_df['Target'] = summary_display_df['Target Constraint']
        summary_display_df['Achieved Volume'] = summary_display_df['Total Volume (tonnes)']
        summary_display_df['Achieved Cost (€)'] = summary_display_df['Total Cost (€)']
        summary_display_df['Avg Price (€/tonne)'] = summary_display_df['Average Price (€/tonne)']

        # Define columns based on constraint type
        if constraint_type == "Volume Constrained":
            summary_display_df.rename(columns={'Target': 'Target Volume'}, inplace=True)
            display_cols = ['Year', 'Target Volume', 'Achieved Volume', 'Achieved Cost (€)', 'Avg Price (€/tonne)']
        else: # Budget Constrained
             summary_display_df.rename(columns={'Target': 'Target Budget (€)'}, inplace=True)
             display_cols = ['Year', 'Target Budget (€)', 'Achieved Cost (€)', 'Achieved Volume', 'Avg Price (€/tonne)']

        # Add volume columns for allocated types
        for proj_type in plot_types_present:
             col_name = f'Volume {proj_type.capitalize()}'
             if col_name in summary_display_df.columns:
                  display_cols.append(col_name)
                  # Format volume column (integer)
                  summary_display_df[col_name] = summary_display_df[col_name].map('{:,.0f}'.format)

        # Format numeric columns for display
        for col in ['Target Budget (€)', 'Achieved Cost (€)', 'Avg Price (€/tonne)']:
             if col in summary_display_df.columns: summary_display_df[col] = summary_display_df[col].map('€{:,.2f}'.format) # Add Euro sign
        for col in ['Target Volume', 'Achieved Volume']:
              if col in summary_display_df.columns: summary_display_df[col] = summary_display_df[col].map('{:,.0f}'.format)

        # Display the formatted dataframe
        st.dataframe(summary_display_df[display_cols].set_index('Year'))

        # --- Raw Allocation Data Section ---
        st.subheader("Detailed Allocation Data")
        if st.checkbox("Show raw project allocations by year", key="show_raw", value=False): # Default to unchecked
             if not portfolio_df.empty:
                 display_portfolio_df = portfolio_df.copy()
                 # Format numbers in the detailed table
                 display_portfolio_df['volume'] = display_portfolio_df['volume'].map('{:,.2f}'.format)
                 display_portfolio_df['price'] = display_portfolio_df['price'].map('€{:,.2f}'.format)
                 display_portfolio_df['cost'] = display_portfolio_df['cost'].map('€{:,.2f}'.format)
                 st.dataframe(display_portfolio_df[['year', 'project name', 'type', 'volume', 'price', 'cost']].sort_values(by=['year', 'project name']))
             else:
                 st.info("No detailed allocation data to display.")

    # --- Message if projects selected but no allocation occurred ---
    elif selected_projects and not portfolio_data_list and allocation_warnings:
         st.error("Projects were selected, but no allocations could be made. Please check warnings above and project data (volume/price) for the selected years.")
    # --- Message if no projects were selected ---
    elif df_upload and not selected_projects:
         st.warning("Please select at least one project in Step 2 to build the portfolio.")

# --- End of Script ---
