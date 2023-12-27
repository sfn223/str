import streamlit as st

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import orbit
from orbit.eda import eda_plot
from orbit.models import KTR, LGT, DLT
from orbit.diagnostics.plot import plot_predicted_data, plot_predicted_components

from sklearn.preprocessing import MaxAbsScaler

from snowflake.snowpark import Session

st.set_option('deprecation.showPyplotGlobalUse', False)

connection_parameters ={
    "account": "xpunefu-dataquest_apps",
    "user": "ABHILASH",
    "role": "sysadmin",
    "password": "Abhids@2024",
    "database": "MMM_POC",
    "schema": "MMM_SCHEMA"
}

session = Session.builder.configs(connection_parameters).create()

tables = session.sql("select table_name from information_schema.tables where table_type = 'BASE TABLE' order by table_name").to_pandas()["TABLE_NAME"].to_list()
table = st.selectbox("Select a table: ", tables)

df = None
if table is not None:
    # load dataframe from snowflake
    data = session.table(table).to_pandas()
    data = data[:-37]
    data['month'] = pd.to_datetime(data.month)
    # columns = list(df.columns)
    st.table(data.head())


# date_col = st.selectbox('Date column', columns)
# response_col = st.selectbox('Response column', columns)

if data is not None:
    col1, col2 = st.columns(2)

    with col1:
        eda_button = st.button("EDA")
    with col2:
        orbit_ml = st.button("Orbit ML")
    # with col3:
    #     dlt_button = st.button("DLT")
    # with col4:
    #     lgt_button = st.button("LGT")

    if eda_button:
        st.write("working")

        eda_plot.wrap_plot_ts(data, date_col='month', var_list=['month', 'sales_total', 'inflation_rate', 'cpi', 'm1', 'm2', 'cli', 'bci', 'cci'])
        st.pyplot()

    if orbit_ml:
        with st.spinner("training"):
            data_train = data[:-24]
            data_test = data[-24:]

            ktr = KTR(
                response_col='sales_total', date_col='month',
                regressor_col=['cpi'],
                seasonality=12,
            )
            ktr.fit(df=data_train)

            ktr_df = ktr.predict(df=data_test, decompose=True)

            dlt = DLT(
                response_col='sales_total', date_col='month',
                regressor_col=['cpi'],
                seasonality=12,
            )
            dlt.fit(df=data_train)

            dlt_df = dlt.predict(df=data_test, decompose=True)

            lgt = LGT(
                response_col='sales_total',
                date_col='month',
                regressor_col=['cpi'],
                seasonality=12,
                seed=8888,
            )
            lgt.fit(df=data_train)

            lgt_df = lgt.predict(df=data_test, decompose=True)

        with col1:
            st.write("KTR")
            st.dataframe(ktr_df)

            st.write("DLT")
            st.dataframe(dlt_df)

            st.write("LGT")
            st.dataframe(lgt_df)

        with col2:
            st.write("KTR")
            plot_predicted_data(
                training_actual_df=data_train,
                predicted_df=ktr_df,
                date_col=ktr.date_col,
                actual_col=ktr.response_col,
                test_actual_df=data_test
            )
            st.pyplot()

            plot_predicted_components(
                ktr_df,
                date_col='month',
                plot_components=['prediction', 'trend', 'seasonality', 'regression']
            )
            st.pyplot()

            st.write("DLT")
            plot_predicted_data(
                training_actual_df=data_train,
                predicted_df=dlt_df,
                date_col=dlt.date_col,
                actual_col=dlt.response_col,
                test_actual_df=data_test
            )
            st.pyplot()

            plot_predicted_components(
                ktr_df,
                date_col='month',
                plot_components=['prediction', 'trend', 'seasonality', 'regression']
            )
            st.pyplot()

            st.write("LGT")
            plot_predicted_data(
                training_actual_df=data_train,
                predicted_df=lgt_df,
                date_col=lgt.date_col,
                actual_col=lgt.response_col,
                test_actual_df=data_test
            )
            st.pyplot()

            plot_predicted_components(
                ktr_df,
                date_col='month',
                plot_components=['prediction', 'trend', 'seasonality', 'regression']
            )
            st.pyplot()
