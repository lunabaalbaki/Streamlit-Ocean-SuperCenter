import streamlit as st
import plotly_express as px
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("sales_str.csv")
df.info()

st.title("Ocean SuperCenter")


nav= st.sidebar.radio("Navigation", ["Home", "Visualization", "Prediction"])
if nav == "Home":
    st.write("In this streamlit app we're going to use the Ocean SuperCenter dataset to build a tool that visualizes features among datasets and to predict a certain feature (Rating) in this case." )
    st.image("/Users/user/Desktop/python/nathalia-rosa-rWMIbqmOxrY-unsplash.jpg",height=10)

    st.write("City Mart Supermarket is a Myanmar's leading supermarket, offering a wide range of international and premium products and fresh food.")





if nav == "Visualization":

    #load data
    df=pd.read_csv("sales_str.csv", encoding= 'unicode_escape')


    product_1 = list(df["Productline"].unique())

    product_profit = []
    for i in product_1:
        x = df[df["Productline"] == i]
        sums = sum(x.Profit)
        product_profit.append(sums)

    data = pd.DataFrame({"product_1": product_1, "product_profit": product_profit})
    new_index = (data["product_profit"].sort_values(ascending=False)).index.values
    sorted_data = data.reindex(new_index)
    trace1 = go.Bar(
                    x = sorted_data.product_1,
                    y = sorted_data.product_profit,
                    name = "Gross Profit by Product Line",
                    marker = dict(color = '#5ce0a0',line = dict(color="rgb(2,65,0)",width=1)))
    data = [trace1]
    layout = dict(title = "Profit by Product Line")
    fig5 = go.Figure(data = data, layout = layout)
    st.plotly_chart(fig5)

    df_cat2= df.groupby(['Branch'], as_index=False)['Profit'].sum()
    fig2 = px.bar(df_cat2, x='Branch', y='Profit', title='Profit Per Branch')
    st.plotly_chart(fig2)

    df_cat6= df.groupby(['City'], as_index=False)['Profit'].sum()
    fig6 = px.bar(df_cat6, x='City', y='Profit', title='Profit Per City')
    fig6.update_traces(marker_color='#5ce0a0')
    st.plotly_chart(fig6)



    df2=df[df['Customer type']=='Member'].iloc[:10,:]
    pie_list=list(df.Profit)
    labels=df2['Payment']
    fig3={
        "data":[
            {
                "values":pie_list,
                "labels":labels,
                "domain": {"x": [.2, 1]},
                "name": "Profit by Customer Type ",
                "hoverinfo":"label+percent+name",
                "hole": .4,
                "type": "pie"
                },],
                "layout":{
                "title":"Payment Preference by Members",
                "annotations":[
                {
                    "font":{"size":17},
                    "showarrow": False,
                    "text": "Types of Payment",
                    "x": 0.75,
                    "y": 0.5
                    },
                    ]
                    }
                    }
    st.plotly_chart(fig3)


    df3=df[df['Customer type']=='Normal'].iloc[:10,:]
    pie_list=list(df.Profit)
    labels=df3['Payment']
    fig4={
        "data":[
            {
                "values":pie_list,
                "labels":labels,
                "domain": {"x": [.2, 1]},
                "name": "Profit by Customer Type ",
                "hoverinfo":"label+percent+name",
                "hole": .4,
                "type": "pie"
                },],
                "layout":{
                "title":"Payment Preference by Non-Members",
                "annotations":[
                {
                    "font":{"size":17},
                    "showarrow": False,
                    "text": "Types of Payment",
                    "x": 0.75,
                    "y": 0.5
                    },
                    ]
                    }
                    }
    st.plotly_chart(fig4)


    if st.button('Visualize the Dataset'):
        st.write(df.head(5))


    #Check nb of rows and Columns
    data_shape=st.radio(" ",('Rows', 'Columns'))
    if data_shape=='Rows':
        st.text("Number of Rows")
        st.write(df.shape[0])
    if data_shape=='Columns':
        st.text("Number of Columns")
        st.write(df.shape[1])


    # 8. Get Overall Statistics
    if st.checkbox("Summary of The Dataset"):
        st.write(df.describe(include='all'))


if nav == "Prediction":



    from sklearn.model_selection import train_test_split
    df2=pd.read_csv("rate_.csv")
    df2['Age'] = df2['Age'].astype(int)
    df2['Rating'] = df2['Rating'].astype(int)


    X = np.array(df2['Age']).reshape(-1,1)
    y=df2['Rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lr = LinearRegression()
    lr.fit(X,np.array(y))



    st.header("Know the Rating based on Age")
    val = st.number_input("Enter Age",0,100,step = 1)
    val = np.array(val).reshape(1,-1)
    pred =lr.predict(val)[0]

    if st.button("Predict"):
        st.success(f"Predicted Rating is {round(pred)} /10.0")



#footer
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Made in ",
        image('https://avatars3.githubusercontent.com/u/45109972?s=400&v=4',
              width=px(25), height=px(25)),
        "by ",
        link("https://github.com/lunabaalbaki?tab=repositories", "Luna Baalbaki"),
        br(),
        link("mailto:lab28@mail.aub.edu", "Send me an Email")

    ]
    layout(*myargs)

if __name__ == "__main__":
    footer()
