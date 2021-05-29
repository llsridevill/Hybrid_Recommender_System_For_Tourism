from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from keras.utils.np_utils import to_categorical
from numpy import dot
from numpy.linalg import norm
import math
import spacy
import datetime
app = Flask(__name__)

global filename
global X, Y, X1
global X_train, X_test, y_train, y_test
global tfidf_vectorizer
global accuracy
locations = []

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

textdata = []
labels = []
tour_ratings_arr = {}
tour_sent_arr = {}
ratings_arr = []
sent_arr = []
choosen_restaurants=""

location_details={
    "Charminar": ["The Charminar constructed in 1591, is a monument and mosque located in Hyderabad, Telangana, India. The landmark has become known globally as a symbol of Hyderabad and is listed among the most recognised structures in India", "https://goo.gl/maps/4kRRgqngRo4jJeJY6","17.3616","78.4747"],
    "Golkonda Fort": ["Golconda Fort, also known as Golkonda, is a fortified citadel and an early capital of the Qutb Shahi dynasty, located in Hyderabad, Telangana, India. Because of the vicinity of diamond mines, especially Kollur Mine, Golconda flourished as a trade centre of large diamonds, known as the Golconda Diamonds","https://goo.gl/maps/jnF6zwQzyuoPeB7q8","17.3833", "78.4011"],
    "Chowmahalla Palace": ["Chowmahalla Palace or Chowmahallat is the palace of the Nizams of Hyderabad State in Hyderabad, Telangana, India. It was the seat of the Asaf Jahi dynasty and was the official residence of the Nizams of Hyderabad while they ruled their state. ", "https://goo.gl/maps/YiPgdzveZRb7CaDj8","17.3578", "78.4717"],
    "Mecca Masjid": ["Makkah Masjid or Mecca Masjid, is a congregational mosque in Hyderabad, India. It is one of the largest mosques in India with a capacity of 10,000 people. The mosque was built between the 16th and 17th centuries, and is a state-protected monument","https://goo.gl/maps/1nUwrgYavpcbzzEi6","17.3604", "78.4736"],
    "Salar Jung Museum": ["The Salar Jung Museum is an art museum located at Dar-ul-Shifa, on the southern bank of the Musi River in the city of Hyderabad, Telangana, India. It is one of the three National Museums of India","https://goo.gl/maps/XArZWFdFs4Su1kKg6","17.3713", "78.4803"],
    "Inorbit Mall": ["Inorbit Mall at Hitech city, Hyd is one of famous shopping complexes in the city. It rooms a vast space of parking for four wheeler and two wheelers as well. It consists of five floors that includes famous branded stores for clothing & accessories, food stalls, electronics, cosmetics and many more. It also has games for children under five YO. There is also a sort of game/entertainment zone for adults known as \"Dialouge in the Dark\" which is an excellent experience that one need to have","https://goo.gl/maps/pBA2Z1vd56dFicdw6","17.4344", "78.3866"],
    "Forum Sujana Mall": ["A lifestyle mall for fashion, entertainment and dining in Hyderabad.","https://goo.gl/maps/1QJC9T4jkxJYScgj9","17.4843", "78.3889"],
    "Sarath City Capital Mall": ["Sarath City Capital Mall is situated in Hyderabad's technology corridor - Hi-Tech City, Gachibowli - Miyapur Road. Sarath City Capital Mall has everything from Prestigious International brands to Local Niche brands that serve the customer in various ways. The mall features AMB Cinemas, a luxury 7 screen multiplex co-owned by Asian Cinemas and Mahesh Babu, the Telugu cinema superstar. SKI capital, Tridom, Thalariya, and SKY zone are the gaming and entertainment zone for everyone. Sarath city capital mall is currently the biggest mall in Hyderabad","https://goo.gl/maps/Wd2DAuyp7jfi6hJ87","17.4576", "78.3639"],
    "GVK One Mall": ["The mall has 70 branded stores and its anchor store Shoppers Stop spans 60,000 sq ft (5,600 m2) in three levels.[3] The mall has a six-screen INOX multiplex housed in it. The mall is established and promoted by GVK Power and Infrastructure limited.[4] The mall also has KFC, Starbucks, Chai Point, and Hard Rock Cafe outlets.","https://goo.gl/maps/18p93wHPTFnSbRBb7","17.4193", "78.4485"],
    "Prasad's IMAX": ["Family complex containing a state-of-the-art cinema, a video game arcade, dining and shopping.","https://goo.gl/maps/7ccas1z1tMfxD54K9","17.4130", "78.4657"],
    "Birla Mandir": ["Birla Mandir in Hyderabad is dedicated to Lord Venkateshwara of Tirumala Tirupati Temple. It is said that a whopping quantity of white marble, which is almost 2000 tonnes, was brought from Rajasthan for building this temple. Located on the top of a hill of 280 feet, also known as Naubat Pahad, Birla Temple stands magnificently, drawing admiring glances from every passerby.","https://goo.gl/maps/YDnZAwJPXxfkBCUP9","17.4062","78.4691"],
    "Sanghi Temple": ["Sanghi Temple, located at Sanghi Nagar in Telangana in India, is about 35 km from Hyderabad city. The sacred Raja Gopuram, which is very tall, can be seen from several kilometers away. The temple complex is located on the top of Paramanand Giri hill, which attracts a number of devotees","https://goo.gl/maps/dVLMJPKajFB6EirT9","17.2668", "78.6760"],
    "Chilkur Balaji Temple": ["Chilkur Balaji Temple, popularly known as \"Visa Balaji Temple\", is an ancient Hindu temple of Lord Balaji on the banks of Osman Sagar in Rangareddy District. It is one of the oldest temples in Hyderabad Dist earlier now in Rangareddy Dist. built during the time of Madanna and Akkanna, the uncles of Bhakta Ramadas","https://goo.gl/maps/zjaM5n8ET2xWJP1x8","17.3587", "78.2988"],
    "Sri Peddamma Temple": ["Sri Peddamma Thalli Temple or Peddamma Gudi is an Hindu temple located at Jubilee Hills in Hyderabad.[1][2] It is very famous during the festive season of Bonaalu.","https://goo.gl/maps/1m8NPFJDWeqN5v8L9","17.4520", "78.4782"],
    "Jagannath Temple":["The Jagannath Temple in Hyderabad, India is a modern temple built by the Odia community of the city of Hyderabad dedicated to the Hindu God Jagannath. The temple located near Banjara hills Road no.12 in Hyderabad is famous for its annual Rathyatra festival attended by thousands of devotees","https://goo.gl/maps/MXijdoWy2EdFU94K7","17.4151", "78.4262"],
    "Ramoji Film City": ["Ramoji Film City is an integrated film studio complex located in Hyderabad, India. Spread over 1,666 acres, it is the largest integrated film city in the world and as such has been certified by the Guinness World Records as the largest studio complex in the world","https://goo.gl/maps/AbR377CFLkaQ2Ujp6","17.4435","78.4607"],
    "Wonderla": ["Amusement park with water rides, roller coasters, kiddie areas & a huge ferris wheel atop a tower.","https://goo.gl/maps/YgNeNkcXpXogUCrj9","17.2173", "78.5285"],
    "Ocean Park": ["Sprawling waterpark & recreational area with a variety of rides, slides & pools.","https://goo.gl/maps/vfXGebmBERX32cJT6","17.3897", "78.3290"],
    "Sky Zone": ["Sky Zone Hyderabad is India's first indoor trampoline park. Attractions include Freestyle Jump, Foam Zone, Drop Zone, SkySlam, SkyLadder, SkyJoust, SkyHoops, Ultimate Dodgeball, SkyLine, Warped Wall and Glow. Best Place for Birthday Parties and Events, Corporate Get together, Team Outings, Birthday Party Venues and more.","https://goo.gl/maps/DEW3p9SkoNjwmnbB7","17.3976", "78.3087"],
    "Snow World": ["Snow World is an amusement park located in Hyderabad, Telangana, India within an area of about 2 acres. Located beside Indira Park and along the Hussain Sagar lake","https://goo.gl/maps/VLsVjMQCqBFv7UgS7", "17.4146","78.4809"],
    "Shilparamam": ["Shilparamam is an arts and crafts village located in Madhapur, Hyderabad, Telangana, India. The village was conceived with an idea to create an environment for the preservation of traditional crafts. There are ethnic festivals round the year","https://goo.gl/maps/pJy1A4xE1QFBb9Hu9","17.4526","78.3783"],
    "Nehru Zoological Park": ["Nehru Zoological Park is a zoo located near Mir Alam Tank in Hyderabad, Telangana, India. It is one of the most visited destinations in Hyderabad.","https://goo.gl/maps/DAntEi35hTDoEwpB7","17.3507", "78.4513"],
    "Hussain Sagar Lake": ["Hussain Sagar is a heart-shaped lake in Hyderabad, Telangana, built by Ibrahim Quli Qutb Shah in 1563. It is spread across an area of 5.7 square kilometres and is fed by the River Musi. A large monolithic statue of the Gautama Buddha, erected in 1992, stands on Gibraltar Rock in the middle of the lake","https://goo.gl/maps/iF2eSKBQbRE8Hb949","17.4239","78.4738"],
    "Qutub Shahi Tombs":["The Qutub Shahi Tombs are located in the Ibrahim Bagh, close to the famous Golconda Fort in Hyderabad, India. They contain the tombs and mosques built by the various kings of the Qutub Shahi dynasty.","https://goo.gl/maps/FDf6j6xepFZmC5sz7","17.3963", "78.3932"],
    "JalaVihar WaterPark": ["Jalavihar is a waterpark located in Hyderabad, Telangana, within an area of about 12.5 acres. Located beside Sanjeevaiah Park and along the Hussain Sagar lake, the park was inaugurated on 20 May 2007.","https://goo.gl/maps/phfYMfPJ5H327m116","17.4326", "78.4648"],


}

sid = SentimentIntensityAnalyzer()

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens


def getCuisine(name, df):
    cuisine = ''
    for i in range(len(df)):
        hname = df[i, 0]
        if hname == name:
            cuisine = df[i, 4]
            break
    return cuisine.strip()


def extractProfile():
    filename = 'static/Dataset/Hyderabad Tourist Dataset.csv'
    ratings_arr.clear()
    sent_arr.clear()
    textdata.clear()
    labels.clear()
    locations.clear()
    metadata = pd.read_csv('static/Dataset/Restaurant names and Metadata.csv')
    metadata = metadata.values
    dataset = pd.read_csv(filename,encoding='iso-8859-1',nrows=5000)
    for i in range(len(dataset)):
        user = dataset.iloc[i]['Reviewer']
        msg = dataset.iloc[i]['Review']
        temp = str(msg)
        if temp != 'nan':
            label = float(dataset.iloc[i]['Rating'])
            ratings_arr.append(label)
            loc = dataset.iloc[i]['Restaurant']
            msg = str(msg)
            msg = msg.strip().lower()
            if label >= 3:
                labels.append(0)
            else:
                labels.append(1)
            clean = cleanPost(msg+" "+getCuisine(loc,metadata))
            textdata.append(clean)
            locations.append(loc)
            sentiment_dict = sid.polarity_scores(temp)
            negative = sentiment_dict['neg']
            positive = sentiment_dict['pos']
            neutral = sentiment_dict['neu']
            compound = sentiment_dict['compound']
            result = ''
            if compound >= 0.05 :
                result = 'Positive'
            elif compound <= - 0.05 :
                result = 'Negative'
            else :
                result = 'Neutral'
            sent_arr.append(result)
            #text.insert(END,user+" Gave Rating "+str(label)+"  & Sentiments "+str(result)+" to location "+str(loc)+" with reviews as "+str(temp)+"\n\n")
    extractContentFeatures()


def extractContentFeatures():
    global Y, X, X1
    global tfidf_vectorizer
    global X_train, X_test, y_train, y_test
    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords)
    tfidf = tfidf_vectorizer.fit_transform(textdata).toarray()
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    #print(df.shape)
    temp = df
    df = df.values
    X = df[:, 0:df.shape[1]]
    X1 = df[:, 0:df.shape[1]]
    Y = np.asarray(labels)
    #le = LabelEncoder()
    #Y = le.fit_transform(Y)
    Y = to_categorical(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    trainDeepLearning()


def trainDeepLearning():
    global Y, X
    global X_train, X_test, y_train, y_test
    cnn_model = Sequential()
    cnn_model.add(Dense(512, input_shape=(X_train.shape[1],)))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(512))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(2))
    cnn_model.add(Activation('softmax'))
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(cnn_model.summary())
    acc_history = cnn_model.fit(X, Y, epochs=3, validation_data=(X_test, y_test))
    print(cnn_model.summary())
    predict = cnn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    acc_history = acc_history.history
    acc_history = acc_history['accuracy']
    acc_history = acc_history[2]


def predict(choosen_restaurants):
    output = []
    c=0
    review = choosen_restaurants
    review = review.lower()
    review = review.strip().lower()
    review = cleanPost(review)
    testArray = tfidf_vectorizer.transform([review]).toarray()
    testArray = testArray[0]
    similarity = 0
    user_recommend = 'Sorry! Unable to recommend for given desired services'
    for i in range(len(X1)):
        recommend = dot(X1[i], testArray)/(norm(X1[i])*norm(testArray))
        if recommend > 0:
            similarity = recommend
            user_recommend = str(locations[i])+" Given Rating "+str(ratings_arr[i])
            if locations[i] not in output:
                output.append(locations[i])
                c=c+1
                if c==5:
                    break
    if similarity == 0:
        output.append("Nothing Found")
    return output

def CalculateDistance(lat1, long1, lat2, long2):
    R = 6373.0
    lat1 = math.radians(lat1)
    lon1 = math.radians(long1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(long2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = abs(math.sin(dlat / 2) * 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) * 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def sortbydistance(placeresult):
    place_code = {}
    for i in range(len(placeresult)):
        place_code[i] = placeresult[i]
    dist = []
    for i in range(0, len(placeresult)):
        a = []
        for j in range(0, len(placeresult)):
            lat1 = float(location_details.get(placeresult[i])[2])
            long1 = float(location_details.get(placeresult[i])[3])
            lat2 = float(location_details.get(placeresult[j])[2])
            long2 = float(location_details.get(placeresult[j])[3])
            d = CalculateDistance(lat1, long1, lat2, long2)
            a.append(d)
        dist.append(a)
    for i in range(len(dist)):
        for j in range(len(dist)):
            print(dist[i][j], end=" ")
        print()
    def path_len(path):
        return sum(dist[i][j] for i, j in zip(path, path[1:]))
    to_visit = set(range(len(dist)))
    state = {(i, frozenset([0, i])): [0, i] for i in range(1, len(dist[0]))}
    for _ in range(len(dist) - 2):
        next_state = {}
        for position, path in state.items():
            current_node, visited = position
            for node in to_visit - visited:
                new_path = path + [node]
                new_pos = (node, frozenset(new_path))
                if new_pos not in next_state or path_len(new_path) < path_len(next_state[new_pos]):
                    next_state[new_pos] = new_path
        state = next_state
    answer = []
    shortest = min((path for path in state.values()), key=path_len)
    for i in shortest:
        answer.append(place_code[i])
    return answer


def extractPlaceProfile():
    ratings_arr.clear()
    textdata.clear()
    labels.clear()
    locations.clear()
    sent_arr.clear()
    # text.delete('1.0', END)
    filename = 'static\Dataset\Hyd_tourdata.csv'
    dataset = pd.read_csv(filename, encoding='iso-8859-1', nrows=5000)
    for i in range(len(dataset)):
        # user = dataset.iloc[i]['username']
        msg = dataset.iloc[i]['Review']
        temp = str(msg)
        if temp != 'nan':
            label = float(dataset.iloc[i]['Rating'])

            loc = dataset.iloc[i]['location']
            if (loc in tour_ratings_arr):
                tour_ratings_arr[loc].append(label)
            else:
                tour_ratings_arr[loc] = [label]
            msg = str(msg)
            msg = msg.strip().lower()
            if label >= 3:
                labels.append(0)
            else:
                labels.append(1)
            # textdata.append(clean)
            locations.append(loc)
            sentiment_dict = sid.polarity_scores(temp)
            negative = sentiment_dict['neg']
            positive = sentiment_dict['pos']
            neutral = sentiment_dict['neu']
            compound = sentiment_dict['compound']
            result = ''
            if compound >= 0.05:
                result = 'Positive'
            elif compound <= - 0.05:
                result = 'Negative'
            else:
                result = 'Neutral'
            if (loc in tour_sent_arr):
                tour_sent_arr[loc].append(result)
            else:
                tour_sent_arr[loc] = [result]


@app.route('/')
def home():
    extractPlaceProfile()
    extractProfile()
    return render_template('blog.html')

@app.route('/result', methods = ['POST','GET'])
def result():
    if request.method=='POST':
        rec=0
        result = request.form
        options=[]
        global choosen_restaurants
        name = result['name']
        email = result['email']
        choosen_type = result.getlist('checklist')
        choosen_rest = result.getlist('place')
        choosen_restaurants=""
        for i in choosen_rest:
            choosen_restaurants+=" "+i
        print(choosen_restaurants)


        if ('historical' in choosen_type):
            rec = 1
            options.append('Charminar')
            options.append('Golkonda Fort')
            options.append('Chowmahalla Palace')
            options.append('Mecca Masjid')
            options.append('Salar Jung Museum')
        if ('shopping' in choosen_type):
            rec = 1
            options.append('Inorbit Mall')
            options.append('Forum Sujana Mall')
            options.append('Sarath City Capital Mall')
            options.append('GVK One Mall')
            options.append('Prasad\'s IMAX')
        if ('spiritual' in choosen_type):
            rec = 1
            options.append('Birla Mandir')
            options.append('Sanghi Temple')
            options.append('Chilkur Balaji Temple')
            options.append('Sri Peddamma Temple')
            options.append('Jagannath Temple')

        if ('fun' in choosen_type):
            rec = 1
            options.append('Ramoji Film City')
            options.append('Wonderla')
            options.append('Ocean Park')
            options.append('Sky Zone')
            options.append('Snow World')

        if('must visit' in choosen_type):
                options.append('Shilparamam')
                options.append('Nehru Zoological Park')
                options.append('Hussain Sagar Lake')
                options.append('Qutub Shahi Tombs')
                options.append('Jala Vihar')

        for place in location_details:
            if place in tour_ratings_arr:
                location_details[place].append(round(sum(tour_ratings_arr[place])/len(tour_ratings_arr[place]),2))
                location_details[place].append(max(set(tour_sent_arr[place]), key = tour_sent_arr[place].count))
            else:
                location_details[place].append(2.3)
                location_details[place].append("Neutral")

        sorted_loc_details = dict(sorted(location_details.items(), key=lambda item: item[1][4], reverse= True))

        return render_template("result.html", result = options, info = sorted_loc_details)

@app.route('/choosenplaces', methods = ['POST','GET'])
def choosenplaces():
    global choosen_restaurants
    print("choosen:   ", choosen_restaurants)
    rec_rest = predict(choosen_restaurants)
    print(rec_rest)
    if request.method == 'POST':
        placeresult = request.form.getlist('places')
        placeresult = sortbydistance(placeresult)
    return render_template("choosenplaces.html", result=placeresult, rest_result=rec_rest)

if __name__ == "__main__":
    app.run()


