import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

n_clusters = 15

st.set_page_config(page_title='時系列予測アプリ（仮）', layout="wide")
st.title('時系列予測アプリ（仮）')

st.sidebar.header('準備')

learning_period = st.sidebar.slider('予測に使用する日数（5日以上 15日以下）', 5, 15, 7, step = 1)


dummy_list = ['1800', '26000', '55000', '53000', '43000', '33000', '30000', '28000', '29000', '34000', '28000', '18000', '17000', '17000', '15000']
dummy_list = dummy_list[:learning_period]
dummy_list = '\n'.join(dummy_list)


sales_num_str = st.sidebar.text_area(f'販売1日目から{learning_period}日目の販売実績', dummy_list, height=25*learning_period+15)
test_values = pd.Series(map(int, sales_num_str.split()))
test_data = test_values.cumsum()

shipment_num = st.sidebar.text_input('出荷数（消化率の計算に使用）', value='870000')

if st.sidebar.button('予測実行'):
    df_sej_60 = pd.read_csv('df_sej_60.csv')
    
    time_series_scaled = []
    for title in set(df_sej_60.タイトル):
        target_df = df_sej_60[df_sej_60.タイトル == title]
        target_series = target_df.累積販売数[:learning_period]
        
        max_target_series = max(target_series)
        min_target_series = min(target_series)
        target_series_scaled = (target_df.累積販売数[:60] - min_target_series) / (max_target_series - min_target_series)
        
        time_series_scaled.append(target_series_scaled.tolist())

    test_X = [target_list[:learning_period] for target_list in time_series_scaled]
    ts = to_time_series_dataset(test_X)

    km = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw")
    labels = km.fit_predict(ts)

    cluster_model = {}

    for label in range(n_clusters):
        target_index = [i for i, x in enumerate(labels) if x == label]

        average_series = pd.Series([0]*60)

        for i in target_index:
            tmp_series = pd.Series(time_series_scaled[i])
            average_series = average_series + (tmp_series/len(target_index))
        cluster_model[label] = average_series
    
    max_test_series = max(test_data)
    min_test_series = min(test_data)
    test_data_scaled = (test_data - min_test_series) / (max_test_series - min_test_series)
    
    test_cluster = int(km.predict(to_time_series_dataset(test_data_scaled[:learning_period])))
    
    predict_series = pd.Series(cluster_model[test_cluster]) * (max_test_series - min_test_series) + min_test_series
    
    digestibility = predict_series/int(shipment_num)*100
    predict_series = predict_series.astype('int64')
    
    for i in predict_series.index:
        if i == 0:
            predict_sales = pd.Series(predict_series[i])
        else:
            predict_sales = predict_sales.append(pd.Series(predict_series[i] - predict_series[i-1]))
    predict_sales = predict_sales.reset_index(drop=True)
    
    st.header('販売数と消化率の遷移')
    
    predict_result = []
    predict_result.append(go.Scatter(x=test_values.index, y=test_values.values, name='販売数実績', yaxis='y1'))
    predict_result.append(go.Scatter(x=predict_sales.index, y=predict_sales.values, name='販売数予測', yaxis='y1'))
    predict_result.append(go.Scatter(x=digestibility.index, y=digestibility.values, name='消化率予測', yaxis='y2'))
    layout = go.Layout(
        xaxis = dict(title='経過日数'),
        yaxis = dict(title='販売数'),
        yaxis2 = dict(title='消化率（％）', showgrid=False, overlaying='y', side='right'),
        hovermode = 'x unified'
    )
    fig_predict_result = go.Figure(data=predict_result, layout=layout)
    fig_predict_result.add_vrect(x0=0, x1=learning_period-1, fillcolor = 'LightSalmon', opacity = 0.5, layer = 'below', line = {'width': 0})
    st.plotly_chart(fig_predict_result, use_container_width=True)
    
    predict_table = pd.DataFrame({
        '経過日数': [f'{i}日目' for i in range(1, 61)], 
        '販売数実績': test_values,
        '販売数予測': predict_sales, 
        '消化率予測': digestibility
    })
    
    fig_predict_result.write_html('predict_graph.html')
    predict_table.to_csv('predict_table.csv', index=False, encoding='shift-jis')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col3:
        st.download_button('グラフのダウンロード', open('predict_graph.html', 'br'), 'predict_graph.html')
    with col4:
        st.download_button('表のダウンロード', open('predict_table.csv', 'br'), 'predict_table.csv')


