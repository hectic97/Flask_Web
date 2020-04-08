from flask import Flask, render_template,url_for,request,redirect
from execution import Text_Generator
import tensorflow as tf
app = Flask(__name__)
# model = tf.keras.Sequential()
model=tf.keras.models.load_model(r'C:\Users\집\Flask\static\model\Generator.h5')
chosen_movies=[]
from execution import Text_Generator


@app.route('/fmr/info')
def info():
	return render_template('info.html')

@app.route('/')
def home():
	return render_template('home.html')


# @app.route('/pandas')

# def index():
    # return render_template('index.html')
#
# def make_read_excel():
#     import pandas as pd
#     # import numpy as np
#     #
#     # writer = pd.ExcelWriter('static/excel_for_flask.xlsx')
#     # df = pd.DataFrame({"col_{}".format(i):list(np.random.randint(0,100,100))for i in range(8)})
#     # df.to_excel(writer, 'sheet1')
#     # writer.save()
#     #
#     # df = pd.read_excel('static/excel_for_flask.xlsx')
#     df = pd.read_csv('static/movie_df.csv')
#
#     return df.to_html()

# @app.route('/submit')
# def main_get(num=None):
#     return render_template('submit.html',num=num)
# @app.route('/calculate',methods=['POST','GET'])
# def calculate(num=None):
#     if request.method=='POST':
#         pass
#     elif request.method=='GET':
#         temp = request.args.get('num')
#         temp = int(temp)
#
#         temp1 = request.args.get('char1')
#         return render_template('submit.html',num=temp,char1=temp1)

@app.route('/fmr', methods=['POST','GET'])
def select():
    if request.method == 'POST':
        print(request.form.getlist('chosen_movie'))
        global chosen_movies
        chosen_movies=request.form.getlist('chosen_movie')
        print(chosen_movies)
        return redirect('/fmr/result')
    return render_template('select.html')

@app.route('/fmr/result')
def results():
    import pandas as pd
    import numpy as np
    a = pd.read_csv('static/txt_info.txt',header=None)
    d = a.iloc[chosen_movies].values
    e = (' ').join(d.squeeze())
    print(chosen_movies)
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    a.loc[850]=e
    cv_a = cv.fit_transform(a.values.squeeze())
    from sklearn.metrics.pairwise import cosine_similarity
    cs = cosine_similarity(cv_a,cv_a)
    f = pd.read_csv('static/movie_df.csv')
    # new_df = f.loc[np.argsort(cs[850])[::-1][len(chosen_movies)+1:len(chosen_movies)+15]].sort_values(by='movie_rating',
    #                                                                                                   ascending=False)[:10]
    indexes=np.argsort(cs[850])[::-1]
    int_chosen_movie = list(map(int, chosen_movies))
    for x in int_chosen_movie:
        indexes=np.delete(indexes, np.where(indexes==x))
    indexes=np.delete(indexes, np.where(indexes==850))
    new_df=f.loc[indexes[:15]].sort_values(by='movie_rating',ascending=False)[:10]
    new_df=new_df.drop(['Unnamed: 0','movie_genre','movie_director','movie_actor'],axis=1)
    new_df = new_df.reset_index(drop=True)
    new_df.index = new_df.index+1
    new_df = new_df.rename(columns={'movie_name': 'MOVIE', 'movie_rating': 'Rating', 'movie_genre_split': 'Genre',
                          'movie_director_split': 'Director', 'movie_actor_split': 'Cast'})

    return'<h1>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;###### TOP 10 MOVIES FOR YOU #####</h1>'+new_df.to_html(justify='center')+"<a href='/fmr'><br>Go Back</a>"
@app.route('/load_model')
def load_model():
    # global model
    # vocab_size = 17673
    # embedding_dim = 1024
    # rnn_units = 512
    # model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[1, None]))
    # print('HELLOOOOOOOOOOOOOOO')
    # model.add(tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
    # model.add(tf.keras.layers.Dense(vocab_size))
    # print('why no load')
    # model.summary()
    # model.load_weights(r'C:\Users\집\Flask\static\model\weights')
    print('loaded')
    return redirect(url_for('poait'))

@app.route('/poait')
def poait():
    # global model
    # import tensorflow as tf
    # vocab_size = 17673
    # embedding_dim = 1024
    # rnn_units = 512
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[1, None]))
    # model.add(
    #     tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
    # model.add(tf.keras.layers.Dense(vocab_size))
    # model.load_weights('../static/model/weights')
    poetry = Text_Generator.main()
    return render_template('aip.html',poetry = poetry)


if __name__ == '__main__':
	app.run(debug=True,threaded= True)
# app.run(debug=False, threaded=False)

"""<div style = "font:italic bold 1.1em/1em 'Georgia',serif;text-align:center;font-size:26px;color:#FFFFFF">
<h1> Movie recommendation system </h1>
</div>"""