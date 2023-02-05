import streamlit
import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


header = streamlit.container()
dataset = streamlit.container()
modelTraining = streamlit.container()


# Created to make sure data is only loaded once
@streamlit.cache
def get_data(filename):
    crop_file_data = pandas.read_csv(filename)
    return crop_file_data


with header:
    streamlit.title('Crop Recommendation System')

with dataset:
    streamlit.write('Enter information about your agricultural environment below')
    # Columns made for input
    c1, c2, c3 = streamlit.columns(3)
    c4, c5, c6, c7 = streamlit.columns(4)
    c8, c9, c10 = streamlit.columns([1, 1, 1])

    streamlit.markdown('***')  # Makes a line to separate the charts

    # Loads data and creates a sorted list of labels of the crops
    crop_data = get_data('Crop_recommendation.csv')
    crop_labels = crop_data['label'].unique()
    sorted_labels = sorted(set(crop_labels))

    # Gets the means of each crop data for descriptive methods
    crop_means = pandas.pivot_table(crop_data, index=['label'], aggfunc='mean')

    # Creates labels and bar chart for crop conditions
    data_label_input = dataset.selectbox("Pick a condition type to see the optimal setting for each crop.",
                                         options=['temperature', 'humidity', 'ph', 'rainfall'])
    streamlit.bar_chart(crop_means, y=data_label_input)

    # Checks if there are any null values in the data
    print(crop_data.isnull().sum())

    streamlit.markdown('***')  # Makes a line to separate the charts

    # creates pie chart below
    crop_label_input = dataset.selectbox("Pick a crop to see the optimal KNP ratio.", options=sorted_labels)

    values = list(range(23))
    key_pairs1 = zip(sorted_labels, values)  # used for converting crop labels to values
    key_pairs2 = zip(values, sorted_labels)  # used for converting numbers to crop labels
    crop_label_to_num = dict(key_pairs1)  # creates dictionary assigning a number to each crop label
    crop_num_to_label = dict(key_pairs2)  # creates dictionary assigning a crop label to each number

    # uses user input to get NPK ratio data
    knp_array = np.array(crop_means.iloc[crop_label_to_num[crop_label_input], :3])
    knp_labels = np.array(['K = ' + str(knp_array[0]), 'N = ' + str(knp_array[1]), 'P = ' + str(knp_array[2])])

    # Creates pie chart on streamlit
    fig, ax = plt.subplots()
    ax.pie(knp_array, labels=knp_labels, labeldistance=1.1)
    streamlit.pyplot(fig)


with modelTraining:
    streamlit.markdown('***')  # Makes a line to separate the charts
    streamlit.header('Model Training Algorithms')
    streamlit.write('Currently in use: Random Forest Classifier Model')

    # Get user input for which crop to grow
    N_input = c1.number_input('N', value=40)
    P_input = c2.number_input('P', value=40)
    K_input = c3.number_input('K', value=40)
    Temp_input = c4.number_input('Temperature', value=30)
    Humidity_input = c5.number_input('Humidity', value=70)
    Ph_input = c6.number_input('Ph Level', value=6)
    Rainfall_input = c7.number_input('Rainfall', value=100)

    # labels data for usage in model training
    X = crop_data.iloc[0:, :7]
    y = crop_data.label

    # changes labels to numbers
    lb = LabelEncoder()
    y = lb.fit_transform(y)

    # Splits data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=125)

    # Random Forest Model
    model_RFC = RandomForestClassifier()
    model_RFC.fit(X_train, y_train)
    train_score1 = model_RFC.score(X_test, y_test).round(5)

    # Creates confusion matrix for Random Forest Model
    predict1 = model_RFC.predict(X_test)
    confusion_matrx_KNC = confusion_matrix(y_test, predict1)
    fig1, ax1 = plt.subplots()
    plt.figure(figsize=(16, 10))
    ax1.set_title('Random Forest Classifier Model - Accuracy: ' + str(train_score1 * 100))
    sns.heatmap(confusion_matrx_KNC, cmap="Greens", annot=True, annot_kws={"size": 11},
                xticklabels=True,
                yticklabels=True, ax=ax1);
    streamlit.write(fig1)

    # Decision Tree Model
    model_DTC = DecisionTreeClassifier()
    model_DTC.fit(X_train, y_train)
    train_score2 = model_DTC.score(X_test, y_test).round(6)

    # Creates confusion matrix for Decision Tree Model
    predict2 = model_DTC.predict(X_test)
    confusion_matrx_KNC = confusion_matrix(y_test, predict2)
    fig2, ax2 = plt.subplots()
    plt.figure(figsize=(16, 10))
    ax2.set_title('Decision Tree Classifier Model - Accuracy: ' + str(train_score2 * 100))
    sns.heatmap(confusion_matrx_KNC, cmap="Greens", annot=True, annot_kws={"size": 11},
                xticklabels=True,
                yticklabels=True, ax=ax2);
    streamlit.write(fig2)

    # Support Vector Machine Model
    model_SVC = SVC()
    model_SVC.fit(X_train, y_train)
    train_score3 = model_SVC.score(X_test, y_test).round(4)

    # Creates confusion matrix for Support Vector Machine Model
    predict3 = model_SVC.predict(X_test)
    confusion_matrx_KNC = confusion_matrix(y_test, predict3)
    fig3, ax3 = plt.subplots()
    plt.figure(figsize=(16, 10))
    ax3.set_title('Support Vector Machine Model - Accuracy: ' + str(train_score3 * 100))
    sns.heatmap(confusion_matrx_KNC, cmap="Greens", annot=True, annot_kws={"size": 11},
                xticklabels=True,
                yticklabels=True, ax=ax3);
    streamlit.write(fig3)

    # KNeighbors Classifier Model
    model_KNC = KNeighborsClassifier()
    model_KNC.fit(X_train, y_train)
    train_score4 = model_KNC.score(X_test, y_test).round(5)

    # Creates confusion matrix for KNeighbors Classifier Model
    predict4 = model_KNC.predict(X_test)
    confusion_matrx_KNC = confusion_matrix(y_test, predict4)
    fig4, ax4 = plt.subplots()
    plt.figure(figsize=(16, 10))
    ax4.set_title('KNeighbors Classifier Model - Accuracy: ' + str(train_score4 * 100))
    sns.heatmap(confusion_matrx_KNC, cmap="Greens", annot=True, annot_kws={"size": 11},
                xticklabels=True,
                yticklabels=True, ax=ax4);
    streamlit.write(fig4)

    # Button that calculates the optimal crop to grow when it is clicked
    run_query = c8.button('Run - Which Crop to Grow')
    if run_query:
        input_data = [[N_input, P_input, K_input, Temp_input, Humidity_input, Ph_input,  Rainfall_input]]
        df_input = pandas.DataFrame(input_data)
        prediction = model_RFC.predict(df_input)
        c9.header(crop_num_to_label[prediction[0]])
