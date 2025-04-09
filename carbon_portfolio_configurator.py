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
    years = st.number_input("How many years should the portfolio span?", min_value=1, max_value=20, value=6, step=1)
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

    # Ask user about constraint type
    constraint_type = st.radio("Are you budget constrained or volume constrained?", ["Volume Constrained", "Budget Constrained"])

    annual_constraints = {}
    if constraint_type == "Volume Constrained":
        st.markdown("**Enter annual purchase volumes (in tonnes):**")
        for year in selected_years:
            annual_constraints[year] = st.number_input(f"Annual purchase volume for {year}:", min_value=0, step=100, value=1000)
    else:
        st.markdown("**Enter annual budget (in €):**")
        for year in selected_years:
            annual_constraints[year] = st.number_input(f"Annual budget for {year} (€):", min_value=0, step=1000, value=10000)

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
        removal_types = ['technical removal', 'natural removal']
        reduction_type = ['reduction']
        all_types = removal_types + reduction_type
        # Create a dictionary to store projects by their type
        project_types = {t: selected_df[selected_df['project type'] == t] for t in all_types}

        # Calculate the target removal percentage for each year based on transition speed
        removal_percentages = []
        for i, year in enumerate(selected_years):
            progress = (i + 1) / years
            factor = 0.5 + 0.1 * transition_speed
            removal_pct = removal_target * (progress ** factor)
            removal_percentages.append(min(removal_pct, removal_target))

        # Define the initial split between technical and natural removals based on user preference
        initial_removal_split = {
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
            annual_limit = annual_constraints.get(year, 0)

            # Dictionary to store the allocated volumes and costs for the current year
            allocated_projects = {}
            total_allocated_volume = 0
            total_allocated_cost = 0
            allocated_removal_volume = 0

            # --- Allocation for Removal Projects ---
            available_removal_projects = pd.concat([project_types.get(rt, pd.DataFrame()) for rt in removal_types]).copy()
            volume_column = f"available volume {year_str}"
            price_column = f"price {year_str}"

            if volume_column in available_removal_projects.columns and price_column in available_removal_projects.columns:
                available_removal_projects = available_removal_projects[pd.notna(available_removal_projects[volume_column]) & (available_removal_projects[volume_column] > 0)].sort_values(by=['priority' if 'priority' in available_removal_projects.columns else price_column, price_column], ascending=[False if 'priority' in available_removal_projects.columns else True, True])
            else:
                st.warning(f"Expected columns {volume_column} or {price_column} not found for removal projects in {year}.")

            target_removal_volume = annual_limit * removal_share

            if constraint_type == "Volume Constrained":
                for _, project in available_removal_projects.iterrows():
                    available_vol = project.get(volume_column, 0)
                    price = project.get(price_column, 0)
                    can_allocate = min(target_removal_volume - allocated_removal_volume, available_vol)
                    if can_allocate > 0:
                        allocated_projects[project['project name']] = {
                            'volume': can_allocate,
                            'price': price,
                            'type': project['project type']
                        }
                        allocated_removal_volume += can_allocate
                        total_allocated_volume += can_allocate
                        if allocated_removal_volume >= target_removal_volume:
                            break
                if allocated_removal_volume < target_removal_volume:
                    st.warning(f"Not enough removal volume available in {year} to meet the target of {target_removal_volume:.2f} tonnes. Achieved: {allocated_removal_volume:.2f} tonnes.")

            elif constraint_type == "Budget Constrained":
                target_budget_removal = annual_limit * removal_share
                allocated_cost_removal = 0
                for _, project in available_removal_projects.iterrows():
                    available_vol = project.get(volume_column, 0)
                    price = project.get(price_column, 0)
                    can_allocate_volume = (target_budget_removal - allocated_cost_removal) / (price + 1e-9)
                    allocate_volume_proj = min(can_allocate_volume, available_vol)
                    cost = allocate_volume_proj * price
                    if cost > 0:
                        allocated_projects[project['project name']] = {
                            'volume': allocate_volume_proj,
                            'price': price,
                            'type': project['project type']
                        }
                        allocated_cost_removal += cost
                        total_allocated_volume += allocate_volume_proj
                        total_allocated_cost += cost
                        if allocated_cost_removal >= target_budget_removal:
                            break
                if allocated_cost_removal < target_budget_removal:
                    st.warning(f"Not enough removal budget available in {year} (€{target_budget_removal:.2f}) to potentially meet the removal target. Spent: €{allocated_cost_removal:.2f}.")

            # --- Allocation for Reduction Projects ---
            available_reduction_projects = project_types.get('reduction', pd.DataFrame()).copy()
            if volume_column in available_reduction_projects.columns and price_column in available_reduction_projects.columns:
                available_reduction_projects = available_reduction_projects[pd.notna(available_reduction_projects[volume_column]) & (available_reduction_projects[volume_column] > 0)].sort_values(by=['priority' if 'priority' in available_reduction_projects.columns else price_column, price_column], ascending=[False if 'priority' in available_reduction_projects.columns else True, True])
            else:
                st.warning(f"Expected columns {volume_column} or {price_column} not found for reduction projects in {year}.")

            target_reduction_volume = annual_limit * reduction_share

            if constraint_type == "Volume Constrained":
                allocated_reduction_volume = 0
                for _, project in available_reduction_projects.iterrows():
                    available_vol = project.get(volume_column, 0)
                    price = project.get(price_column, 0)
                    can_allocate = min(target_reduction_volume - allocated_reduction_volume, available_vol)
                    if can_allocate > 0:
                        allocated_projects[project['project name']] = {
                            'volume': can_allocate,
                            'price': price,
                            'type': 'reduction'
                        }
                        allocated_reduction_volume += can_allocate
                        total_allocated_volume += can_allocate
                        if allocated_reduction_volume >= target_reduction_volume:
                            break
                if allocated_reduction_volume < target_reduction_volume:
                    st.warning(f"Not enough reduction volume available in {year} to meet the target of {target_reduction_volume:.2f} tonnes. Achieved: {allocated_reduction_volume:.2f} tonnes.")

            elif constraint_type == "Budget Constrained":
                target_budget_reduction = annual_limit * reduction_share
                allocated_cost_reduction = 0
                for _, project in available_reduction_projects.iterrows():
                    available_vol = project.get(volume_column, 0)
                    price = project.get(price_column, 0)
                    can_allocate_volume = (target_budget_reduction - allocated_cost_reduction) / (price + 1e-9)
                    allocate_volume_proj = min(can_allocate_volume, available_vol)
                    cost = allocate_volume_proj * price
                    if cost > 0:
                        allocated_projects[project['project name']] = {
                            'volume': allocate_volume_proj,
                            'price': price,
                            'type': 'reduction'
                        }
                        allocated_cost_reduction += cost
                        total_allocated_volume += allocate_volume_proj
                        total_allocated_cost += cost
                        if allocated_cost_reduction >= target_budget_reduction:
                            break
                if allocated_cost_reduction < target_budget_reduction:
                    st.warning(f"Not enough reduction budget available in {year} (€{target_budget_reduction:.2f}) to potentially meet the reduction target. Spent: €{allocated_cost_reduction:.2f}.")

            portfolio[year] = allocated_projects

        # --- Portfolio Analysis and Visualization ---
        composition_by_type = {t: [] for t in all_types}
        avg_prices = []
        yearly_costs = {}
        yearly_volumes = {}

        # Calculate the total volume per type and the average price for each year
        for year in selected_years:
            annual_constraint = annual_constraints[year]
            total_cost = 0
            total_volume = 0
            type_volumes = {t: 0 for t in all_types}
            for project_info in portfolio[year].values():
                type_volumes[project_info['type']] += project_info['volume']
                total_cost += project_info['volume'] * project_info['price']
                total_volume += project_info['volume']

            for t in all_types:
                composition_by_type[t].append(type_volumes[t])

            avg_prices.append(total_cost / (total_volume + 1e-9) if total_volume > 0 else 0)
            yearly_costs[year] = total_cost
            yearly_volumes[year] = total_volume

        st.subheader("Portfolio Composition & Price Over Time")

        # Create a Plotly figure with a secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add bar traces for the volume of each project type
        for type_name in all_types:
            color = '#8BC34A' if type_name == 'technical removal' else '#AED581' if type_name == 'natural removal' else '#C5E1A5'
            fig.add_trace(go.Bar(x=selected_years, y=composition_by_type[type_name], name=type_name.capitalize(), marker_color=color), secondary_y=False)

        # Add a line trace for the average price
        fig.add_trace(go.Scatter(x=selected_years, y=avg_prices, name='Average Price (€/tonne)', marker=dict(symbol='circle'), line=dict(color='#558B2F')), secondary_y=True)

        # Update layout for better aesthetics
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Volume (tonnes)',
            yaxis2_title='Average Price (€/tonne)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            barmode='stack',  # Stack the bar charts for volume composition
            template="plotly_white"  # Use a clean template
        )

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig)

        # Calculate and display the achieved removal percentage in the final year
        final_tech = composition_by_type['technical removal'][-1]
        final_nat = composition_by_type['natural removal'][-1]
        final_total_removal = final_tech + final_nat
        final_total = final_total_removal + composition_by_type['reduction'][-1]
        achieved_removal = (final_total_removal) / (final_total + 1e-9) * 100 if final_total > 0 else 0

        st.markdown(f"**Achieved Removal % in {end_year}: {achieved_removal:.2f}%**")

        st.subheader("Yearly Summary")
        summary_data = {'Year': selected_years}
        if constraint_type == "Volume Constrained":
