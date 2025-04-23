[Portfolio](https://github.com/daluchkin/data-analyst-portfolio) |  [Projects](https://github.com/daluchkin/data-analyst-portfolio/blob/main/projects.md) | [Certificates](https://github.com/daluchkin/data-analyst-portfolio/blob/main/certificates.md) | [Contacts](https://github.com/daluchkin/data-analyst-portfolio#my_contacts)

# Divvy: Bike Sharing Forecast

The Divvy bicycle sharing service in Chicago offers an environmentally friendly and convenient transportation option, supported by a public-private partnership with Lyft. With a broad network of docking stations and a fleet of standard and electric bicycles, Divvy serves a diverse range of users. This project focuses on analyzing ride data to forecast the number of rides and to understand the impact of external factors such as weather conditions and public holidays. By examining these variables, the project aims to identify patterns in usage, determine peak times, and evaluate the effects of different conditions on ride frequency. The insights gained will assist in optimizing operations, planning for demand fluctuations, and enhancing overall service delivery, contributing to the promotion of sustainable urban mobility in Chicago.

> **Attention**
> The data utilized in this project were sourced from the official public data provided by Divvy. All analysis, results, and insights presented in this notebook were conducted and formulated by Dmitry Luchkin. I declare that I am not affiliated with Divvy, Lyft, the City of Chicago, or any associated organizations. The views and interpretations expressed herein are solely my own and do not represent the positions or policies of these entities.

## Notebooks
+ [`01_Divvy_initial_data_exploration.ipynb`](./01_Divvy_initial_data_exploration.ipynb)
+ [`02_Divvy_data_cleaning.ipynb`](./02_Divvy_data_cleaning.ipynb)
+ [`03_Divvy_exploratory_data_analysis.ipynb`](./03_Divvy_exploratory_data_analysis.ipynb)
+ [`04_Divvy_feature_engineering.ipynb`](./04_Divvy_feature_engineering.ipynb)
+ Modeling & Validation:
    + [`05.1_Divvy_modeling_xgboost.ipynb`](./05.1_Divvy_modeling_xgboost.ipynb)
    + [`05.2_Divvy_modeling_rnn.ipynb`](./05.2_Divvy_modeling_rnn.ipynb)
    + [`05.3_Divvy_modeling_arimax.ipynb`](./05.3_Divvy_modeling_arimax.ipynb)
    + [`05.4_Divvy_modeling_mstl.ipynb`](./05.4_Divvy_modeling_mstl.ipynb)
    + [`05.5_Divvy_modeling_ensemble.ipynb`](./05.5_Divvy_modeling_ensemble.ipynb)
+ [`06_Divvy_forecasting.ipynb`](./06_Divvy_forecasting.ipynb)


## Scope of the Project

The scope of this project encompasses the following key activities and analyses related to the Divvy bicycle sharing service in Chicago:

- __Data Collection and Preparation__

    - Gather historical ride data from Divvy, including details such as trip duration, start and end locations, and time of rides.
    - Collect additional data on external factors such as weather conditions (temperature, precipitation, etc.) and public holidays.

- __Exploratory Data Analysis (EDA)__

    - Analyze the collected data to identify patterns and trends in ride usage.
    - Determine peak usage times, popular routes, and any seasonal variations in the data.

- __Impact Analysis__

    - Assess the impact of weather conditions on the number of rides, identifying which weather factors most significantly influence usage.
    - Evaluate the effect of public holidays on ride frequency, including variations in usage patterns during these periods.

- __Forecasting Ride Numbers__

    - Develop and implement predictive models to forecast the number of rides, using historical data and identified influencing factors.
    - Test and validate the accuracy of these models using appropriate statistical and machine learning techniques.

- __Reporting and Visualization__

    - Create detailed reports and visualizations to present findings, including interactive dashboards or visual aids for stakeholders.
    - Document the methodology, results, and conclusions drawn from the analyses.

## Out of Scope
- Analysis of user demographic data, as this information is not included in the available dataset.
- Development of new infrastructure or direct operational changes, as the project is limited to data analysis and recommendations.
- Real-time monitoring and adjustment of services, as the focus is on historical data and predictive modeling rather than live system management.
- This scope outlines the boundaries and focus areas of the project, ensuring that all activities are aligned with the objectives and expected outcomes. It also clarifies what is not included, providing a clear understanding of the project's limitations.

## Goals of the Project

- Analyze ride data to forecast the number of rides.
- Examine the impact of weather conditions on ride frequency.
- Assess the effect of public holidays on the number of rides.
- Identify patterns in usage, including peak times and popular routes.
- Provide insights for optimizing the operational aspects of the service.
- Support planning for demand fluctuations based on predictive analysis.

## Dataset

This dataset is 53 .CSV files containing data points with information about rides in bike sharing service from **January 2020** to **December 2024**.

Each trip is anonymized and includes:

- Trip start day and time
- Trip end day and time
- Trip start station
- Trip end station
- Rider type (Member, Single Ride, and Day Pass)

The data has been processed to remove trips that are taken by staff as they service and inspect the system; and any trips that were below 60 seconds in length (potentially false starts or users trying to re-dock a bike to ensure it was secure).


### Description of Attributes

| Attribute            | Description                                                      |
|----------------------|------------------------------------------------------------------|
| `ride_id`            | Ride ID                                                          |
| `rideable_type`      | Type of a bike                                                   |
| `started_at`         | Trip start day and time                                          |
| `ended_at`           | Trip end day and time                                            |
| `start_station_name` | Trip start station name                                          |
| `start_station_id`   | Trip start station ID                                            |
| `end_station_name`   | Trip end station name                                            |
| `end_station_id`     | Trip end station ID                                              |
| `start_lat`          | Latitude of a start point                                        |
| `start_lng`          | Latitude of a start point                                        |
| `end_lat`            | Latitude of a start point                                        |
| `end_lng`            | Latitude of a start point                                        |
| `member_casual`      | Rider type                                                       |


### Lisense
This data is provided according to the [Divvy Data License Agreement](https://www.divvybikes.com/data-license-agreement) and released on a monthly schedule.


[Portfolio](https://github.com/daluchkin/data-analyst-portfolio) |  [Projects](https://github.com/daluchkin/data-analyst-portfolio/blob/main/projects.md) | [Certificates](https://github.com/daluchkin/data-analyst-portfolio/blob/main/certificates.md) | [Contacts](https://github.com/daluchkin/data-analyst-portfolio#my_contacts)