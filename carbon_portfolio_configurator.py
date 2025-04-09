import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt  # While imported, it's not used in the current plotting logic
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns  # For potentially more appealing styles (not directly used in plotting)

# --- Streamlit App Configuration ---
# Set Streamlit page config to customize the background color using HTML/CSS
st.markdown("""
    <style>
    body {
        background-color: #D9E1C8; /* Light green background color */
    }
    .stApp {
        background-color: #D9E1C8; /* Apply background color to the main app area */
    }
    </style>
""", unsafe_allow_html=True)

# Set the title of the Streamlit application
st.title("🌱 Multi-Year Carbon Portfolio Builder")

# --- Data Input ---
# File uploader widget to allow users to upload a CSV file
df = st.file_uploader("Upload project CSV", type="csv")

# --- Main Logic (Conditional on CSV Upload) ---
if df:
    # Read the uploaded CSV file into a Pandas DataFrame
    data = pd.read_csv(df)

    # --- Project Overview Section ---
    st.subheader("Project Overview")

    # Slider to determine the number of years for the portfolio
    years = st.slider("How many years should the portfolio span?", 1, 6, 6)
    # Calculate the end year based on the starting year (2025) and the number of years
    end_year = 2025 + years - 1
    # Create a list of selected years based on the slider value
    selected_years = list(range(2025, end_year + 1))
    
    # Create a copy of the data for the overview
    overview = data.copy()
    # Identify price columns based on the selected years
    price_cols = [f"price {year}" for year in selected_years if f"price {year}" in data.columns]
    # Calculate the average price across the selected years
    overview['avg_price'] = overview[price_cols].mean(axis=1)
    # Display the project overview DataFrame, including description if available
    if 'description' in overview.columns:
        st.dataframe(overview[['project name', 'project type', 'description', 'avg_price']].drop_duplicates())
    else:
        st.dataframe(overview[['project name', 'project type', 'avg_price']].drop_duplicates())
    
    # --- Portfolio Settings Section ---
    st.subheader("Step 1: Define Portfolio Settings")
    # Dictionary to store the annual purchase volume for each year
    annual_volumes = {}
    # Input fields for annual purchase volume for each selected year
    for year in selected_years:
        annual_volumes[year] = st.number_input(f"Annual purchase volume for {year}:", min_value=0, step=100, value=1000)

    # Slider to set the target total removal percentage for the final year
    removal_target = st.slider(f"Target total removal % for year {end_year}", 0, 100, 80) / 100
    # Slider to control the transition speed of the removal target over the years
    transition_speed = st.slider("Transition Speed (1 = Slow, 10 = Fast)", 1, 10, 5)
    # Slider to set the preference for natural vs. technical removals
    removal_preference = st.slider("Removal Preference (1 = Natural, 10 = Technical)", 1, 10, 5)

    # --- Project Selection and Prioritization Section ---
    st.subheader("Step 2: Select and Prioritize Projects")
    # Get unique project names from the data
    project_names = data['project name'].unique().tolist()
    # Multiselect widget to allow users to select projects for the portfolio
    selected_projects = st.multiselect("Select projects to include in the portfolio:", project_names)
    # Multiselect widget to allow users to mark favorite projects for a priority boost
    favorite_projects = st.multiselect("Select your favorite projects (will get a +10% priority boost):", selected_projects)

    # --- Portfolio Allocation Logic (Conditional on Project Selection) ---
    if selected_projects:
        # Create a DataFrame containing only the selected projects
        selected_df = data[data['project name'].isin(selected_projects)].copy()

        # Apply priority boost to favorite projects
        if favorite_projects:
            selected_df['priority'] = selected_df.apply(
                lambda row: row['priority'] + 10 if row['project name'] in favorite_projects and pd.notna(row['priority']) and row['priority'] < 100 else row['priority'],
                axis=1
            )
            # Ensure the priority does not exceed 100%
            selected_df['priority'] = selected_df['priority'].clip(upper=100)

        # Define the types of projects
        types = ['technical removal', 'natural removal', 'reduction']
        # Create a dictionary to store projects by their type
        project_types = {t: selected_df[selected_df['project type'] == t] for t in types}

        # Calculate the target removal percentage for each year based on transition speed
        removal_percentages = []
        for i, year in enumerate(selected_years):
            progress = (i + 1) / years
            factor = 0.5 + 0.1 * transition_speed
            removal_pct = removal_target * (progress ** factor)
            removal_percentages.append(min(removal_pct, removal_target))

        # Define the split between technical and natural removals based on user preference
        category_split = {
            'technical removal': removal_preference / 10,
            'natural removal': 1 - (removal_preference / 10)
        }

        # Dictionary to store the allocated portfolio for each year
        portfolio = {year: {} for year in selected_years}
        # List to store any broken allocation rules or issues
        broken_rules = []

        # --- Allocation Loop for Each Year ---
        for year_idx, year in enumerate(selected_years):
            year_str = f"{year}"
            removal_share = removal_percentages[year_idx]
            reduction_share = 1 - removal_share
            annual_volume = annual_volumes.get(year, 0)

            # Dictionary to store the allocated volumes for each project in the current year
            volumes = {}
            total_allocated = 0

            # --- Allocation Loop for Each Project Type ---
            for category in types:
                # Determine the share of the annual volume for the current category
                if category == 'reduction':
                    category_share = reduction_share
                else:
                    category_share = removal_share * category_split[category]

                # Get projects of the current category with available volume for the current year
                category_projects = project_types[category].copy()
                category_projects = category_projects[category_projects.get(f"available volume {year_str}", 0) > 0]

                # If no projects are available for the current category and year, add a broken rule
                if category_projects.empty:
                    broken_rules.append(f"No available volume for {category} in {year}.")
                    continue

                allocated_category_volume = 0

                # --- Priority-Based Allocation ---
                if 'priority' in category_projects.columns:
                    # Separate projects with and without a priority value
                    priority_projects = category_projects[category_projects['priority'].notna()].copy()
                    remaining_projects = category_projects[category_projects['priority'].isna()].copy()

                    # Allocate volume to priority projects based on their priority
                    for _, row in priority_projects.iterrows():
                        priority = row['priority'] / 100.0  # Convert percentage to fraction
                        target_volume = annual_volume * category_share * priority
                        max_available = row.get(f"available volume {year_str}", 0)
                        vol = min(target_volume, max_available)
                        volumes[row['project name']] = {
                            'volume': int(vol),
                            'price': row.get(f'price {year_str}', 0),
                            'type': category
                        }
                        allocated_category_volume += vol

                    # Allocate remaining volume equally among non-priority projects
                    num_remaining = len(remaining_projects)
                    if num_remaining > 0:
                        remaining_category_volume = annual_volume * category_share - allocated_category_volume
                        if remaining_category_volume > 0:
                            share_per_remaining = remaining_category_volume / num_remaining
                            for _, row in remaining_projects.iterrows():
                                max_available = row.get(f"available volume {year_str}", 0)
                                vol = min(share_per_remaining, max_available)
                                volumes[row['project name']] = {
                                    'volume': int(vol),
                                    'price': row.get(f'price {year_str}', 0),
                                    'type': category
                                }
                                allocated_category_volume += vol
                # --- Equal Allocation if No Priority Column ---
                else:
                    # If 'priority' column doesn't exist, allocate equally among projects
                    num_projects = len(category_projects)
                    if num_projects > 0:
                        share_per_project = annual_volume * category_share / num_projects
                        for _, row in category_projects.iterrows():
                            max_available = row.get(f"available volume {year_str}", 0)
                            vol = min(share_per_project, max_available)
                            volumes[row['project name']] = {
                                'volume': int(vol),
                                'price': row.get(f'price {year_str}', 0),
                                'type': category
                            }
                            allocated_category_volume += vol

                total_allocated += allocated_category_volume

            # Scale the allocated volumes if the total exceeds the annual volume
            if total_allocated > 0:
                scale_factor = annual_volume / total_allocated
                for v in volumes.values():
                    v['volume'] = int(v['volume'] * scale_factor)
            else:
                broken_rules.append(f"No available projects in {year}, cannot allocate volume.")

            # Store the allocated volumes for the current year in the portfolio dictionary
            portfolio[year] = volumes

        # --- Portfolio Analysis and Visualization ---
        composition_by_type = {t: [] for t in types}
        avg_prices = []

        # Calculate the total volume per type and the average price for each year
        for year in selected_years:
            annual_volume = annual_volumes[year]
            totals = {t: 0 for t in types}
            total_cost = 0
            for data in portfolio[year].values():
                totals[data['type']] += data['volume']
                total_cost += data['volume'] * data['price']

            for t in types:
                composition_by_type[t].append(totals[t])

            avg_prices.append(total_cost / annual_volume if annual_volume > 0 else 0)

        st.subheader("Portfolio Composition & Price Over Time")

        # Create a Plotly figure with a secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add bar traces for the volume of each project type
        fig.add_trace(go.Bar(x=selected_years, y=composition_by_type['technical removal'], name='Technical Removal', marker_color='#8BC34A'), secondary_y=False)
        fig.add_trace(go.Bar(x=selected_years, y=composition_by_type['natural removal'], name='Natural Removal', marker_color='#AED581'), secondary_y=False)
        fig.add_trace(go.Bar(x=selected_years, y=composition_by_type['reduction'], name='Reduction', marker_color='#C5E1A5'), secondary_y=False)

        # Add a line trace for the average price
        fig.add_trace(go.Scatter(x=selected_years, y=avg_prices, name='Average Price (€)', marker=dict(symbol='circle'), line=dict(color='#558B2F')), secondary_y=True)

        # Update layout for better aesthetics
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Volume',
            yaxis2_title='Average Price (€)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            barmode='stack',  # Stack the bar charts for volume composition
            template="plotly_white"  # Use a clean template
        )

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig)

        # Calculate and display the achieved removal percentage in the final year
        final_tech = composition_by_type['technical removal'][-1]
        final_nat = composition_by_type['natural removal'][-1]
        final_total = final_tech + final_nat + composition_by_type['reduction'][-1]
        achieved_removal = (final_tech + final_nat) / final_total if final_total > 0 else 0

        st.markdown(f"**Achieved Removal % in {end_year}: {achieved_removal * 100:.2f}%**")

        # Display any broken allocation rules or warnings
        if broken_rules:
            st.warning("One or more constraints could not be fully satisfied:")
            for msg in broken_rules:
                st.text(f"- {msg}")

        # Checkbox to show the raw project allocations
        if st.checkbox("Show raw project allocations"):
            full_table = []
            for year, projects in portfolio.items():
                for name, info in projects.items():
                    full_table.append({
                        'year': year,
                        'project name': name,
                        'type': info['type'],
                        'volume': info['volume'],
                        'price': info['price'],
                        'cost': info['volume'] * info['price']
                    })
            st.dataframe(pd.DataFrame(full_table))
