import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import json
from datetime import date
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
import textwrap

# Connect to SQLite

# conn = sqlite3.connect('../data/enron.db')

database_path = 'data/enron.db'

@st.cache_data
def load_topic_wc(source_table='scipy_labels'):
    """Connects to the database and loads the topic and embedding data."""
    try:
        conn = sqlite3.connect(database_path) # connect to sqlite database
        df_scipy = pd.read_sql(f'SELECT *  FROM {source_table}', conn)

        conn.close()
        # Convert embedding_json string back to a list
        df_scipy['word_cloud'] = df_scipy['word_cloud_json'].apply(json.loads)
        df_scipy.drop(columns=['word_cloud_json'], inplace=True)


        return df_scipy
    
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
        return pd.DataFrame()



@st.cache_data
def load_topic_analysis(source_table='dbscan_topic_breakdown_analysis', topic=1):
    """Connects to the database and loads slices for analysis."""
    try:
        conn = sqlite3.connect(database_path) # connect to sqlite database
        df_analysis = pd.read_sql(f'''SELECT
                date_short, from_addr, person_box, sub_mailbox, sum(email_count) as email_count, sum(unique_email_count) as unique_email_count
                FROM {source_table}
                WHERE topic = {topic}
                group by date_short, from_addr, person_box, sub_mailbox
            
                ''', conn)

        conn.close()

        return df_analysis
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
        return pd.DataFrame()
    


# --- Database Connection and Data Loading ---
@st.cache_data
def load_data():
    """Connects to the database and loads the topic and embedding data."""
    try:
        conn = sqlite3.connect(database_path) # connect to sqlite database
        df_embed = pd.read_sql(f'''SELECT hdbscan_clusters_roberta,reduced_labels_scipy_cut,
                               reduced_labels_dbscan_cut, embedding_roberta_json FROM embeddings
        where hdbscan_clusters_roberta > -1 ''', conn)
        conn.close()
        # Convert embedding_json string back to a list
        df_embed['embedding'] = df_embed['embedding_roberta_json'].apply(json.loads)
        
        return df_embed
        st.success("Data loaded successfully.") 
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
        return pd.DataFrame()

# df_embed = load_data()
df_embed = None

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the sentence-transformers model."""
    # Using a RoBERTa-based model suitable for sentence embeddings
    model_name = 'sentence-transformers/all-roberta-large-v1' 
    return SentenceTransformer(model_name)

model = load_model()



# Placeholder: Load your actual DataFrame here
# df =  pd.read_pickle('..\data\df_clean_2001_roberta_pick.pkl.gz', compression='gzip')
# df = df_clean_2001_pick

# def get_top_tfidf_words(texts, top_n=10):
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     vectorizer = TfidfVectorizer(stop_words="english")
#     X = vectorizer.fit_transform(texts)
#     indices = X.mean(axis=0).A1.argsort()[::-1][:top_n]
#     features = vectorizer.get_feature_names_out()
#     return {features[i]: X.mean(axis=0).A1[i] for i in indices}

def make_wordcloud(words_freq):
    wc = WordCloud(width=400, height=250, background_color="white")
    wc.generate_from_frequencies(words_freq)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    return fig

@st.cache_resource
def representative_email(source_table='embeddings',topic_label='reduced_labels_dbscan_cut', topic_value=1):
    try:
        conn = sqlite3.connect(database_path) # connect to sqlite database
        filtered_emails = pd.read_sql(f'''SELECT cleaned_email_body, hdbscan_clusters_roberta_prob  FROM
        {source_table}
    WHERE
        {topic_label} = {topic_value}
    order by hdbscan_clusters_roberta_prob DESC
    limit 2
        ''', conn)# order by hdbscan_clusters_roberta_prob DESC
        conn.close()

        return filtered_emails
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
        return pd.DataFrame()
    
@st.cache_resource
def topic_email(source_table='embeddings',topic_label='reduced_labels_dbscan_cut', topic_value=1):
    try:
        conn = sqlite3.connect(database_path) # connect to sqlite database
        filtered_emails = pd.read_sql(f'''SELECT email_body, cleaned_email_body,hdbscan_clusters_roberta_prob FROM
        {source_table}
    WHERE
        {topic_label} = {topic_value}
    order by hdbscan_clusters_roberta_prob DESC
    
        ''', conn)# order by hdbscan_clusters_roberta_prob DESC
        conn.close()

        return filtered_emails
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
        return pd.DataFrame()

def make_analysis_charts(df):
    
    df['date_short_month']  = pd.to_datetime(df['date_short']).dt.strftime('%b')
    df['person_box'] = df['person_box'].str.upper()
    # Ensure all months are present, even with zero emails
    all_months = [date(2001, m, 1).strftime('%b') for m in range(1, 13)]
    df_trend = df.groupby(['date_short_month','sub_mailbox'])['email_count'].sum().reset_index()
    df_trend.columns = ['date_short_month','sub_mailbox' ,'email_count']
    
    month_map = {date(2001, m, 1).strftime('%b'): m for m in range(1, 13)}
    df_trend['month_num'] = df_trend['date_short_month'].map(month_map)
    df_trend = df_trend.sort_values('month_num')

    # st.dataframe(df_trend)
    # Define the correct order of the months
    month_order = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]

    fig_line = px.bar(
        df_trend, 
        x='date_short_month', 
        y='email_count', 
        title='Total Email Count by Month',
        color='sub_mailbox',
        labels={'date_short_month': 'Month', 'email_count': 'Total Email Count'},
        category_orders={'date_short_month': month_order} # This is the key line
    )
    fig_line.update_layout(xaxis_title="Month", yaxis_title="Total Email Count")
    # st.plotly_chart(fig_line, use_container_width=True)

    # # --- Pie Charts for Aggregated Data ---
    # st.header("Email Distribution")
    # # col1, col2, col3 = st.columns(3)

    # # Pie chart for person_box
    # # with col1:
    # st.subheader("Emails by Person Box")
    df_person_box = df[~df['sub_mailbox'].isin(['sent', 'sent_items', '_sent_mail'])].groupby('person_box')['email_count'].sum().reset_index()
    fig_person = px.pie(
        df_person_box, 
        values='email_count', 
        names='person_box', 
        title='Distribution by Person Box'
    )
    # st.plotly_chart(fig_person, use_container_width=True)

    # Pie chart for from_addr
    # with col2:
    # st.subheader("Emails by Sender Address")
    df_from_addr = df.groupby('from_addr')['email_count'].sum().reset_index()
    fig_from = px.pie(
        df_from_addr, 
        values='email_count', 
        names='from_addr', 
        title='Distribution by Sender Address'
    )
    # st.plotly_chart(fig_from, use_container_width=True)

    # Pie chart for sub_mailbox
    # with col3:
    # st.subheader("Emails by Sub-Mailbox")
    df_sub_mailbox = df.groupby('sub_mailbox')['email_count'].sum().reset_index()
    fig_sub = px.pie(
        df_sub_mailbox, 
        values='email_count', 
        names='sub_mailbox', 
        title='Distribution by Sub-Mailbox'
    )
    # st.plotly_chart(fig_sub, use_container_width=True)

    return fig_line, fig_person, fig_from, fig_sub



st.title("Cluster Topic Explorer")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["similarity_search", "reduced_labels_scipy_cut", "reduced_labels_dbscan_cut", "hdbscan_labels(Uncut)","more email samples"])#, "hdbscan_clusters_roberta"])


with tab1:
    # --- Streamlit App Setup ---
    # st.title("Topic-Based Similarity Search")

    # st.subheader("Vector Database Loading")

    # if st.button("Reload Data", on_click=lambda: load_data.clear()):
    #     df_embed = load_data()
    #     st.success("Data reloaded successfully.")

   
        
    # --- User Input ---
    st.subheader("Topic-Based Similarity Search")
    st.markdown("Enter a mock email to find the most similar topics in the database.")
    query_string = st.text_area(
        "Enter search sentences (preferably email length):",
        ''' I'm still a bit uncomfortable with the proposed structure for the Fishtail LLC. The valuation report 
from Chase Securities seems overly optimistic, and I'm concerned that the $2 million equity 
injection from our investor isn't truly independent. As you know, we need to maintain a minimum of 
3% of the total capital to avoid consolidation, and the report doesn't provide sufficient evidence of 
this. Could we discuss ways to legally justify the 3% requirement, or perhaps explore alternative 
funding that isn't so reliant on this specific transaction? I'm worried that if this deal falls through, it 
will have a significant impact on our reported earnings and debt levels.''',
    )

    if st.button("Search",key='button1'):
        # st.write(query_string.split('\n'))
        if df_embed is None:
            df_embed = load_data()

        if not query_string:
            st.warning("Please enter some text to perform a search.")
        else:
            with st.spinner("Processing search..."):
                # Encode the user query
                query_embeddings = model.encode(query_string)

                # Prepare corpus embeddings for calculation
                # Convert the list of lists into a single NumPy array
                corpus_embeddings = np.vstack(df_embed['embedding'].values)

                # --- Calculate Cosine Similarity ---
                # Calculate similarity between the query and all corpus embeddings
                # st.write(query_embeddings.dtype)
                # st.write(corpus_embeddings.dtype)

                query_embeddings = query_embeddings.astype('float32')
                corpus_embeddings = corpus_embeddings.astype('float32')

                similarity_scores = util.cos_sim(query_embeddings, corpus_embeddings)

                # Add the similarity scores to the DataFrame
                df_embed['similarity'] = similarity_scores[0]

                # --- Aggregate Similarity by Topic ---
                # Group by topic and calculate the mean similarity
                topic_similarity_scipy = df_embed.groupby('reduced_labels_scipy_cut')['similarity'].agg(['mean', 'count']).reset_index()
                topic_similarity_scipy.rename(
                    columns={
                        'reduced_labels_scipy_cut': 'Scipy_cut Topic',
                        'mean': 'Average Similarity Score',
                        'count': 'email Count'
                    },
                    inplace=True
                )

                topic_similarity_dbscan = df_embed.groupby('reduced_labels_dbscan_cut')['similarity'].agg(['mean', 'count']).reset_index()
                topic_similarity_dbscan.rename(
                    columns={
                        'reduced_labels_dbscan_cut': 'Dbscan_cut Topic',
                        'mean': 'Average Similarity Score',
                        'count': 'email Count'
                    },
                    inplace=True
                )

                topic_similarity_hdb = df_embed.groupby('hdbscan_clusters_roberta')['similarity'].agg(['mean', 'count']).reset_index()
                topic_similarity_hdb.rename(
                    columns={
                        'hdbscan_clusters_roberta': 'Hdbscan Topic',
                        'mean': 'Average Similarity Score',
                        'count': 'email Count'
                    },
                    inplace=True
                )

                # --- Output Results ---
                st.subheader("Top Topics by Aggregate Similarity")
                
                # Sort topics by average similarity in descending order
                top_topics_scipy = topic_similarity_scipy.sort_values(by='Average Similarity Score', ascending=False)
                top_topics_dbscan = topic_similarity_dbscan.sort_values(by='Average Similarity Score', ascending=False)
                top_topics_hdb = topic_similarity_hdb.sort_values(by='Average Similarity Score', ascending=False)
                
                # Display the result as a DataFrame
                st.subheader("Scipy Cut Topics")
                st.dataframe(top_topics_scipy)
                st.subheader("DBSCAN Cut Topics")
                st.dataframe(top_topics_dbscan)
                st.subheader("HDBSCAN Topics")  
                st.dataframe(top_topics_hdb)
                # # Optional: Display a visual of the scores
                # st.bar_chart(top_topics.set_index('Scipy Topic')['Average Similarity Score'])

                





with tab2:
    

    label = "topic"
    st.header(label)
    df_scipy= load_topic_wc(source_table='scipy_labels')
    options_scipy_cut = sorted(df_scipy[label].unique())
    selected_scipy = st.selectbox("Select topic from scipy cut (~500)", options_scipy_cut)
    filtered = df_scipy[df_scipy[label] == selected_scipy].reset_index()


    # st.write(filtered['word_cloud'])
    st.subheader("Topic Wordcloud")
    st.pyplot(make_wordcloud(filtered['word_cloud'][0]))

   
    filtered_emails_scipy = representative_email(source_table='embeddings', topic_label='reduced_labels_scipy_cut', topic_value=selected_scipy)


    st.subheader("Most representative email")
    # st.subheader('-------  sample 1  ----')
    st.markdown('<p style="font-size:25px;">------------  sample 1  ------------</p>', unsafe_allow_html=True)
    st.write(filtered_emails_scipy['cleaned_email_body'][0])
    st.markdown('<p style="font-size:25px;">------------  sample 2  ------------</p>', unsafe_allow_html=True)
    if len(filtered_emails_scipy) > 1:
        st.write(filtered_emails_scipy['cleaned_email_body'][1])


    df_scipy_analysis= load_topic_analysis(source_table='scipy_topic_breakdown_analysis', topic=selected_scipy)

    fig_line1, fig_person1, fig_from1, fig_sub1 = make_analysis_charts(df_scipy_analysis)

        # --- Email Count Trend by Date (Jan to Dec) ---
    st.header("Email Count Trend by Date")
    st.plotly_chart(fig_line1, use_container_width=True,key="scipy_trend")
        # --- Pie Charts for Aggregated Data ---
    st.header("Email Distribution")
    # st.subheader("Emails by Person Box")
    st.plotly_chart(fig_person1, use_container_width=True,key="scipy_person")
    # Pie chart for person_box
    # st.subheader("Emails by Sender Address")
    st.plotly_chart(fig_from1, use_container_width=True,key="scipy_from")
    # Pie chart for sub_mailbox
    # st.subheader("Emails by Sub-Mailbox")
    st.plotly_chart(fig_sub1, use_container_width=True,key="scipy_sub")

with tab3:
    label = "topic"
    st.header(label)
    df_dbscan= load_topic_wc(source_table='dbscan_labels')
    options_dbscan = sorted(df_dbscan[label].unique())
    selected_dbscan = st.selectbox("Select topic from dbscan_cut (~790)", options_dbscan)
    filtered = df_dbscan[df_dbscan[label] == selected_dbscan].reset_index()


    # st.write(filtered['word_cloud'])
    st.subheader("Topic Wordcloud")
    st.pyplot(make_wordcloud(filtered['word_cloud'][0]))

    filtered_emails_dbscan= representative_email(source_table='embeddings', topic_label='reduced_labels_dbscan_cut', topic_value=selected_dbscan)

    st.subheader("Most representative email")
    # st.subheader('-------  sample 1  ----')
    st.markdown('<p style="font-size:25px;">------------  sample 1  ------------</p>', unsafe_allow_html=True)
    st.write(filtered_emails_dbscan['cleaned_email_body'][0])
    st.markdown('<p style="font-size:25px;">------------  sample 2  ------------</p>', unsafe_allow_html=True)
    if len(filtered_emails_dbscan) > 1:
        st.write(filtered_emails_dbscan['cleaned_email_body'][1])



    df_dbscan_analysis= load_topic_analysis(source_table='dbscan_topic_breakdown_analysis', topic=selected_dbscan)

    fig_line2, fig_person2, fig_from2, fig_sub2 = make_analysis_charts(df_dbscan_analysis)

    #     # --- Email Count Trend by Date (Jan to Dec) ---
    st.header("Email Count Trend by Date")
    st.plotly_chart(fig_line2, use_container_width=True,key="dbscan_trend")
        # --- Pie Charts for Aggregated Data ---
    st.header("Email Distribution")
    # st.subheader("Emails by Person Box")
    st.plotly_chart(fig_person2, use_container_width=True,key="dbscan_person")
    # Pie chart for person_box
    # st.subheader("Emails by Sender Address")
    st.plotly_chart(fig_from2, use_container_width=True,key="dbscan_from")
    # Pie chart for sub_mailbox
    # st.subheader("Emails by Sub-Mailbox")
    st.plotly_chart(fig_sub2, use_container_width=True,key="dbscan_sub")

with tab4:
    label = "topic"
    st.header(label)
    df_hdbscan= load_topic_wc(source_table='hdbscan_labels')
    options_hdbscan = sorted(df_hdbscan[label].unique())
    selected_hdbscan = st.selectbox("Select topic from hdbscan (~4700)", options_hdbscan)
    filtered = df_hdbscan[df_hdbscan[label] == selected_hdbscan].reset_index()


    # st.write(filtered['word_cloud'])
    st.subheader("Topic Wordcloud")
    st.pyplot(make_wordcloud(filtered['word_cloud'][0]))

    filtered_emails_hdbscan= representative_email(source_table='embeddings', topic_label='hdbscan_clusters_roberta', topic_value=selected_hdbscan)

    st.subheader("Most representative email")
    # st.subheader('-------  sample 1  ----')
    st.markdown('<p style="font-size:25px;">------------  sample 1  ------------</p>', unsafe_allow_html=True)
    st.write(filtered_emails_hdbscan['cleaned_email_body'][0])
    st.markdown('<p style="font-size:25px;">------------  sample 2  ------------</p>', unsafe_allow_html=True)
    if len(filtered_emails_hdbscan) > 1:
        st.write(filtered_emails_hdbscan['cleaned_email_body'][1])
    # st.markdown('<p style="font-size:25px;">------------  sample 3  ------------</p>', unsafe_allow_html=True)
    # if len(filtered_emails_hdbscan) > 2:
    #     st.write(filtered_emails_hdbscan['email_body'][2])
    # st.markdown('<p style="font-size:25px;">------------  sample 4  ------------</p>', unsafe_allow_html=True)
    # if len(filtered_emails_hdbscan) > 3:
    #     st.write(filtered_emails_hdbscan['email_body'][3])



    df_hdbscan_analysis= load_topic_analysis(source_table='hdbscan_topic_breakdown_analysis', topic=selected_hdbscan)

    
    fig_line3, fig_person3, fig_from3, fig_sub3 = make_analysis_charts(df_hdbscan_analysis)

        # --- Email Count Trend by Date (Jan to Dec) ---
    st.header("Email Count Trend by Date")
    st.plotly_chart(fig_line3, use_container_width=True,key="hdbscan_trend")
        # --- Pie Charts for Aggregated Data ---
    st.header("Email Distribution")
    # st.subheader("Emails by Person Box")
    st.plotly_chart(fig_person3, use_container_width=True,key="hdbscan_person")
    # Pie chart for person_box
    # st.subheader("Emails by Sender Address")
    st.plotly_chart(fig_from3, use_container_width=True,key="hdbscan_from")
    # Pie chart for sub_mailbox
    # st.subheader("Emails by Sub-Mailbox")
    st.plotly_chart(fig_sub3, use_container_width=True,key="hdbscan_sub")
    
with tab5:

    st.header("Retrieve Emails for topics")
    options_field = ['hdbscan_clusters_roberta', 'reduced_labels_dbscan_cut', 'reduced_labels_scipy_cut']
    
    selected_field = st.selectbox("Cluster Groups", options_field)
    cluster_number=st.number_input("Input cluster number",min_value=0,max_value=4800)

    # Custom CSS for word wrap in dataframe
    custom_css = """
            <style>
                /* Target the cells (td) and headers (th) within the dataframe */
                .dataframe td, .dataframe th {
                    white-space: pre-wrap;   /* This is the key property for text wrapping */
                    vertical-align: top;     /* Aligns text to the top of the cell */
                    font-size: 14px;         /* Optional: Adjust font size */
                    word-break: break-word;  /* Ensures long words are broken if they exceed cell width */
                }
            </style>
            """
    if st.button("Search",key='button2'):
        with st.spinner("Processing search..."):
            # Inject the custom CSS
            
            filtered_emails_all = topic_email(source_table='embeddings', topic_label=selected_field, topic_value=cluster_number)
            # filtered_emails_all['cleaned_email_body'] = filtered_emails_all['cleaned_email_body'].apply(
            #                     lambda x: textwrap.fill(x, width=900)
            #                 )
            # st.markdown(custom_css, unsafe_allow_html=True)
            # st.dataframe(filtered_emails_all[['cleaned_email_body']], use_container_width=True)
            
            # Function to handle moving to the next record

            # Iterate through each row in the DataFrame
            for index, row in filtered_emails_all.iterrows():
                # Use st.container() to group elements for a clean visual separation
                if index > 30:
                    break
                with st.container(border=True):
                    st.subheader(f"Email {index + 1}")

                    # Display subject (optional column)
                    # if 'email_subject' in filtered_emails_all.columns:
                    #     st.markdown(f"**Subject:** {textwrap.fillrow['email_subject']}")

                    # Display the long body text using st.markdown
                    # This ensures the text wraps naturally within the Streamlit page width
                    # st.markdown(f"**Body Content:**")
                    
                    # Display the content within a markdown code block for pre-formatted wrapping
                    st.markdown(textwrap.fill(row['email_body'],width=500))
                
                # Add a little space between records
                # st.markdown("---")
                            