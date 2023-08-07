```python
import gdown
```


```python
!gdown --id 1KmoZYmPzxrrtR4a3mSfPIumbDwf_Mrpw > /dev/null 2>&1
!gdown --id 19vK08esg9hQtD8JB-9jm7Qkck7nZ-ZqR > /dev/null 2>&1
# !gdown --id 1pbHI6jeqpajD4lkIhv65uhw4pacq-FUp > /dev/null 2>&1
```

## About the Dataset
This a YouTube comments dataset scrapped using the [YouTube Data API](https://developers.google.com/youtube/v3/docs/comments). Currently the dataset consists of roughly **1.12M** comments. The dataset can be accessed [here](https://drive.google.com/drive/folders/1-9OnYbFuiSA0M7skOYche5erzZ4Ba5Nz?usp=sharing).
<br><br>
The dataset currently has text in _English, Hindi and Hinglish_.
Please refer to the following information for an overview of the columns and the corresponding data stored within them. [Link](https://github.com/aatmanvaidya/Sentiment-Analysis-of-Online-Harassment-Towards-Women-Wrestlers/blob/scraper/attributes.txt).


```python
!pip install wordcloud > /dev/null 2>&1
```


```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from collections import Counter
import re
from wordcloud import WordCloud
```


```python
%%time
df = pd.read_feather(r'comments_cleaned_feather.feather')
```

    CPU times: user 1.65 s, sys: 865 ms, total: 2.51 s
    Wall time: 2.79 s
    


```python
df.head()
```






  <div id="df-c10dff30-7adb-4eae-abb0-d3484bff37ed">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>videoId</th>
      <th>textDisplay</th>
      <th>textOriginal</th>
      <th>authorDisplayName</th>
      <th>authorProfileImageUrl</th>
      <th>authorChannelUrl</th>
      <th>authorChannelId</th>
      <th>canRate</th>
      <th>viewerRating</th>
      <th>likeCount</th>
      <th>publishedAt</th>
      <th>updatedAt</th>
      <th>parentId</th>
      <th>commentId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EBOKYsWUhvI</td>
      <td>Dub maro jaato 😢😢.&lt;br&gt;Tumse tumare ladkiya nhi...</td>
      <td>Dub maro jaato 😢😢.\nTumse tumare ladkiya nhi b...</td>
      <td>HECTOR OF TROY</td>
      <td>https://yt3.ggpht.com/ytc/AOPolaQNP5bd7gNvbAas...</td>
      <td>http://www.youtube.com/channel/UC5G8fjqoiFIqHp...</td>
      <td>{'value': 'UC5G8fjqoiFIqHpKyVeOTsFg'}</td>
      <td>True</td>
      <td>none</td>
      <td>0</td>
      <td>2023-07-06T07:04:02Z</td>
      <td>2023-07-06T07:04:02Z</td>
      <td>None</td>
      <td>UgwbyvIkkAhUdaCFpcp4AaABAg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EBOKYsWUhvI</td>
      <td>Haar gye bechare</td>
      <td>Haar gye bechare</td>
      <td>Geeta Saini</td>
      <td>https://yt3.ggpht.com/8XmOPNrT3Vy3wr0fItlWbaMk...</td>
      <td>http://www.youtube.com/channel/UCsMGRdH3YHrbs2...</td>
      <td>{'value': 'UCsMGRdH3YHrbs21NOraRuyQ'}</td>
      <td>True</td>
      <td>none</td>
      <td>0</td>
      <td>2023-07-03T22:21:37Z</td>
      <td>2023-07-03T22:21:37Z</td>
      <td>None</td>
      <td>Ugyz3OwSXamho91-8I94AaABAg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EBOKYsWUhvI</td>
      <td>Dhamki mili pahalwano ko aur sab manage kr liy...</td>
      <td>Dhamki mili pahalwano ko aur sab manage kr liy...</td>
      <td>Ayaan Chouhan</td>
      <td>https://yt3.ggpht.com/ytc/AOPolaREH2WnrnbD53OI...</td>
      <td>http://www.youtube.com/channel/UC_dWuNh6zydTHI...</td>
      <td>{'value': 'UC_dWuNh6zydTHIRr6hi3Omg'}</td>
      <td>True</td>
      <td>none</td>
      <td>0</td>
      <td>2023-07-03T05:06:33Z</td>
      <td>2023-07-03T05:06:33Z</td>
      <td>None</td>
      <td>Ugyphs1TT1Yoj7MZBVJ4AaABAg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EBOKYsWUhvI</td>
      <td>Jaato pr ye boj rhega ki vo apni hi vyavstha s...</td>
      <td>Jaato pr ye boj rhega ki vo apni hi vyavstha s...</td>
      <td>Kamal</td>
      <td>https://yt3.ggpht.com/ytc/AOPolaTRSm_dEOKj9H82...</td>
      <td>http://www.youtube.com/channel/UCVdGObpHM-IMHB...</td>
      <td>{'value': 'UCVdGObpHM-IMHB_b7_K-7rA'}</td>
      <td>True</td>
      <td>none</td>
      <td>1</td>
      <td>2023-07-02T03:39:50Z</td>
      <td>2023-07-02T03:39:50Z</td>
      <td>None</td>
      <td>UgwnIxpuFAcKsEzffBp4AaABAg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EBOKYsWUhvI</td>
      <td>Pahlwan jante h kuch nhi kr payenge uska</td>
      <td>Pahlwan jante h kuch nhi kr payenge uska</td>
      <td>jagriti tiwari upp</td>
      <td>https://yt3.ggpht.com/ytc/AOPolaRrFnzD2i3N_3rk...</td>
      <td>http://www.youtube.com/channel/UCRkcewHFhxE5Kf...</td>
      <td>{'value': 'UCRkcewHFhxE5KfLHrLx0wpA'}</td>
      <td>True</td>
      <td>none</td>
      <td>0</td>
      <td>2023-06-28T15:10:36Z</td>
      <td>2023-06-28T15:10:36Z</td>
      <td>None</td>
      <td>UgxjuOlDigsmuDu53J54AaABAg</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c10dff30-7adb-4eae-abb0-d3484bff37ed')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>



    <div id="df-eee633ea-783c-44c8-ac6d-fa8969ae71bb">
      <button class="colab-df-quickchart" onclick="quickchart('df-eee633ea-783c-44c8-ac6d-fa8969ae71bb')"
              title="Suggest charts."
              style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>
    </div>

<style>
  .colab-df-quickchart {
    background-color: #E8F0FE;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: #1967D2;
    height: 32px;
    padding: 0 0 0 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: #E2EBFA;
    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: #174EA6;
  }

  [theme=dark] .colab-df-quickchart {
    background-color: #3B4455;
    fill: #D2E3FC;
  }

  [theme=dark] .colab-df-quickchart:hover {
    background-color: #434B5C;
    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
    fill: #FFFFFF;
  }
</style>

    <script>
      async function quickchart(key) {
        const containerElement = document.querySelector('#' + key);
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      }
    </script>

      <script>

function displayQuickchartButton(domScope) {
  let quickchartButtonEl =
    domScope.querySelector('#df-eee633ea-783c-44c8-ac6d-fa8969ae71bb button.colab-df-quickchart');
  quickchartButtonEl.style.display =
    google.colab.kernel.accessAllowed ? 'block' : 'none';
}

        displayQuickchartButton(document);
      </script>
      <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c10dff30-7adb-4eae-abb0-d3484bff37ed button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c10dff30-7adb-4eae-abb0-d3484bff37ed');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df.shape
```




    (1119948, 14)



## Data Cleaning
- removed `^M`
- `\u200b` char (because of devnagiri script)
- header rows appened multiple times
- publishedAt has time like 0,1,2 - remove all that
- convert text column to string


```python
header = np.array(df.columns)
df = df[~np.all(df.values == header, axis=1)]
df = df[~df['videoId'].str.contains(r'\bvideoId\b', case=False, regex=True)]
pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$'
df = df[df['publishedAt'].str.contains(pattern)]
```


```python
df['textOriginal'] = df['textOriginal'].astype(str)
```


```python
duplicate_rows = df[df.duplicated(subset='textOriginal', keep=False)]
```


```python
len(df['textDisplay'].unique())
```




    76258




```python
# Now this is strange, the dataset doesnt have duplicate or empty rows, why is this happening? is it some datatype problem?
# Lets try to do a simple check for it again
```


```python
%%time
count = 0
value_list = []
for index, row in df.iterrows():
    value_list.append(row['textOriginal'])
    count+=1
print(count)
print(len(value_list))
```

    996419
    996419
    CPU times: user 52.3 s, sys: 148 ms, total: 52.4 s
    Wall time: 1min 3s
    


```python
set_value = set(value_list)
print(len(set_value))
```

    76256
    


```python
len(df['videoId'].unique())
```




    148




```python
# This means we are penetrating through the entire dataset. I dont know why pandas is doing that ://
# There is one small problem, there is a \u200b character that is present, we might have to remove that.
```

## Simple Slur List Frequency Count
This is a simple frequency count of how many words from the slur list are found in the data.
I have stored the results in a dictonary called _'slur_counts'_


```python
with open('slur_list.txt', 'r') as file:
    slur_words = [word.strip() for word in file.readlines()]
```


```python
len(slur_words)
```




    506




```python
slur_words_set = set(slur_words)
```


```python
# A dictonary to count the frequency of each slur word identified
slur_counts_beforeRegex = {}
sentence_beforeRegex_list = []
```


```python
%%time
'''
for text in df['textOriginal']:
    words = text.split()
    for word in words:
        if word in slur_words:
            if word in slur_counts:
                slur_counts[word] += 1
            else:
                slur_counts[word] = 1
'''
for text in df['textOriginal']:
    if text is not None:
        words = text.split()
        slur_words_in_text = set(words) & slur_words_set
        if slur_words_in_text:
            if len(sentence_beforeRegex_list)<=100:
                sentence_beforeRegex_list.append([text,slur_words_in_text])
        for word in slur_words_in_text:
            if word in slur_counts_beforeRegex:
                slur_counts_beforeRegex[word] += 1
            else:
                slur_counts_beforeRegex[word] = 1
# Problem with code, in set if a user has used the same word multiple times, set will dismiss that.
```

    CPU times: user 4.85 s, sys: 12.7 ms, total: 4.86 s
    Wall time: 4.94 s
    


```python
sorted_slur_counts_beforeRegex = dict(sorted(slur_counts_beforeRegex.items(), key=lambda item: item[1], reverse=True))
```


```python
# sorted_slur_counts_beforeRegex
# list(sorted_slur_counts_beforeRegex.items())[:8]
```


```python
print(sum(sorted_slur_counts_beforeRegex.values()))
print(len(sorted_slur_counts_beforeRegex.keys()))
```

    22158
    121
    


```python
# sentence_beforeRegex_list
```


```python
english_word_freq_dict = {word: freq for word, freq in sorted_slur_counts_beforeRegex.items() if word.isalpha()}
word_cloud_text = ' '.join([word + ' ' * freq for word, freq in english_word_freq_dict.items()])
```


```python
'''
One thing is clear from this, all the explicit words that are in english are not there in the list,
this means youtube either deletes them or hides them. And the moderation for local vernacular language is weak.
'''
```




    '\nOne thing is clear from this, all the explicit words that are in english are not there in the list,\nthis means youtube either deletes them or hides them. And the moderation for local vernacular language is weak.\n'




```python
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generate_wordcloud(word_count_dict):
    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white')

    # Generate the word cloud from the dictionary
    wordcloud.generate_from_frequencies(word_count_dict)

    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    generate_wordcloud(english_word_freq_dict)
```


    
![png](youtube_comments_analysis_files/youtube_comments_analysis_30_0.png)
    


# Regex
What is a Regex? -> a sequence of characters that specifies a match pattern in text.

**Step 1** - Clean and Preprocess the Data
1.   Remove User Names - `r'@\w+\b'`
2.   Remove URL's and unecessary puntuation marks - `[^\w\s.]|http\S+|www\S+|https\S+`
3.  Remove Double spaces - `r'\s+', ' '`
4.  Remove any leading or trailing spaces - `.strip()`




```python
text = "@user, temp, temp, ;;;;;;;; I hate the offensive and racist comments\n\n\n. They disgust me. Visit https://example.com for more information."
```


```python
def clean_text(text):
    cleaned_text = re.sub(r'@\w+\b|[^\w\s.]|(?:https?|ftp)://\S+|www.\S+', '', text).replace('\n', '').strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text
```


```python
print(clean_text(text))
```

    temp temp I hate the offensive and racist comments. They disgust me. Visit for more information.
    


```python
regexDf = df[['textOriginal']]
```


```python
# clean text
regexDf['textOriginal'] = regexDf['textOriginal'].apply(lambda x: clean_text(x))
```

    <ipython-input-98-547ff038857b>:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      regexDf['textOriginal'] = regexDf['textOriginal'].apply(lambda x: clean_text(x))
    

### Just check the count of words from the slur list ignoring case (upper or lower) of the word. We also look if any word from the slur list is present as a sub-string of any other word in the dataset.

1.   `re.findall()` - searches for all non-overlapping occurrences of a pattern in a given text.
2.   `(?i)` - make matching case-insenstive
3.  `'(' + '|'.join(slur_words) + r')'` - find a pattern that matches any of the slur words


```python
slur_counts_afterRegex = {}
sentence_afterRegex_list = []
```


```python
%%time
for index, row in regexDf.iterrows():
    text = row['textOriginal']
    matches = re.findall(r"(?i)\b(" + '|'.join(slur_words) + r")\b", text)
    # matches = re.findall(r'(?i)(' + '|'.join(slur_words) + r')', text)
    if matches:
        if (len(sentence_afterRegex_list) <= 100) and (text not in sentence_beforeRegex_list):
            sentence_afterRegex_list.append([text, matches])
    for match in matches:
        slur_counts_afterRegex[match] = slur_counts_afterRegex.get(match, 0) + 1
```

    CPU times: user 5min 9s, sys: 715 ms, total: 5min 9s
    Wall time: 5min 14s
    


```python
# slur_counts_afterRegex
```


```python
# prev count - 101382
print(sum(slur_counts_afterRegex.values()))
print(len(slur_counts_afterRegex.keys()))
```

    31763
    211
    


```python
sorted_slur_counts_afterRegex = dict(sorted(slur_counts_afterRegex.items(), key=lambda item: item[1], reverse=True))
print(list(sorted_slur_counts_afterRegex.items())[:8])
```

    [('chutiya', 2418), ('chutiye', 1746), ('chod', 1695), ('mc', 1197), ('bsdk', 1158), ('gaddar', 1069), ('sali', 979), ('harami', 939)]
    


```python
# sentence_afterRegex_list
```


```python
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generate_wordcloud(word_count_dict):
    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white')

    # Generate the word cloud from the dictionary
    wordcloud.generate_from_frequencies(word_count_dict)

    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    generate_wordcloud(sorted_slur_counts_afterRegex)
```


    
![png](youtube_comments_analysis_files/youtube_comments_analysis_44_0.png)
    



```python
# We clearly see that the amount of slur words drastically increase. Now the step ahead would be too create seperate regex for the most used slur words.
# Note - the substrings could also be a part of different words and not necessarily hateful. But given the nature of the words, I feel that chances are low.
```


```python
# substrings are not a good way to
# slur list add - randistan
```

## Create Regex for the most used Slur Words

- Lets start by looking at the word `"chutiya"`


```python
text = "they called me a chhhutiyyaa and I ignored them."
# text = "abcchhutiyeexvy"
```


```python
pattern_w1 = r'ch+h?[uo]t[iy]a?'
# pattern_w1 = r'\bch+h?[uo]t[iy]a?\b'
```


```python
matche_w1 = re.findall(pattern_w1, text, re.IGNORECASE)
```


```python
matche_w1
```




    ['chhhuti']




```python
slur_word_count_w1 = {}
sentences_w1 = []
```


```python
%%time
for index, row in regexDf.iterrows():
    text = row['textOriginal']
    matches = re.findall(pattern_w1, text, re.IGNORECASE)
    if matches:
        if len(sentences_w1)<=100:
            sentences_w1.append([text,matches])
    for match in matches:
        slur_word_count_w1[match] = slur_word_count_w1.get(match, 0) + 1
```

    CPU times: user 50.9 s, sys: 138 ms, total: 51 s
    Wall time: 51.4 s
    


```python
sum(slur_word_count_w1.values())
```




    8010




```python
sorted_slur_word_count_w1 = dict(sorted(slur_word_count_w1.items(), key=lambda item: item[1], reverse=True))
print(list(sorted_slur_word_count_w1.items())[:8])
```

    [('chuti', 5368), ('Chuti', 1021), ('choti', 443), ('chutia', 344), ('chuty', 171), ('chutya', 168), ('chhoti', 164), ('Chutya', 84)]
    


```python
# sentences_w1
```


```python
# Okay wow, the number of words that are similar to `chutiya` increase drastically.
```


```python
# one problem with this is, we dont know what the sentence might be directed to, to women? to the govt? to brijbhusan?
# or just some guy reply to a person in a comment?
```

- now lets look at `chod`


```python
text_2 = "they called me a chhut, chod but I ignored them, hmpf"
```


```python
pattern_w2 = r'ch+[ou]+[dt]'
```


```python
match_w2 = re.findall(pattern_w2, text_2, re.IGNORECASE)
```


```python
match_w2
```




    ['chhut', 'chod']




```python
%%time
slur_word_count_w2 = {}
for index, row in regexDf.iterrows():
    text = row['textOriginal']
    matches_2 = re.findall(pattern_w2, text, re.IGNORECASE)
    for match in matches_2:
        slur_word_count_w2[match] = slur_word_count_w2.get(match, 0) + 1
```

    CPU times: user 50.8 s, sys: 134 ms, total: 50.9 s
    Wall time: 51.3 s
    


```python
sum(slur_word_count_w2.values())
```




    24013




```python
sorted_slur_word_count_w2 = dict(sorted(slur_word_count_w2.items(), key=lambda item: item[1], reverse=True))
print(list(sorted_slur_word_count_w2.items())[:8])
```

    [('chod', 8599), ('chut', 7072), ('chud', 1457), ('chot', 1437), ('chhod', 1313), ('Chut', 1311), ('Choud', 353), ('chhot', 343)]
    

- now lets do this for `bsdk`


```python
text_3 = "they called me a bsssdk, I left"
```


```python
pattern_3 = r'bs+d+k'
matches = re.findall(pattern_3, text_3, re.IGNORECASE)
```


```python
matches
```




    ['bsssdk']




```python
%%time
slur_word_count_4 = {}
sentence_w3 = []
for index, row in regexDf.iterrows():
    text = row['textOriginal']
    matches = re.findall(pattern_3, text, re.IGNORECASE)
    if matches:
        sentence_w3.append([text, matches])
    for match in matches:
        slur_word_count_4[match] = slur_word_count_4.get(match, 0) + 1
```

    CPU times: user 50.8 s, sys: 118 ms, total: 50.9 s
    Wall time: 51.3 s
    


```python
sum(slur_word_count_4.values())
```




    1719




```python
sorted_slur_word_count_4 = dict(sorted(slur_word_count_4.items(), key=lambda item: item[1], reverse=True))
print(list(sorted_slur_word_count_4.items())[:8])
```

    [('bsdk', 1219), ('Bsdk', 453), ('Bssdk', 32), ('BSDK', 15)]
    

- now lets try for a hindi word -

some links that help me understand this better
- https://stackoverflow.com/questions/41356013/how-to-detect-if-a-string-contains-hindi-devnagri-in-it-with-character-and-wor
- https://stackoverflow.com/questions/14859957/regular-expressions-with-indian-characters
- https://unicode.org/charts/PDF/U0900.pdf


```python
# TO-DO
```

## Lets now look at the impact regex has had on counting the slur words in the data

The slur list has 506 words consisting of different languages.

|        **Word**       | **Before Regex** |     **After Regex**     |
|:---------------------:|:----------------:|:-----------------------:|
| Simple Count of Words |      25,393      | 101,382 (just  english) |
|        chutiya        |       2,647      |          9,289          |
|          chod         |       1,788      |          12,749         |
|          bsdk         |       1,147      |          1,836          |


### Looking at use of slur_words accross the timeline.


```python
timeDf = df[['videoId','textOriginal','authorChannelId', 'likeCount', 'publishedAt', 'updatedAt','parentId', 'commentId']]
```


```python
# Convert time from  ISO 8601 to a datetime object
timeDf['NewDateTime'] = pd.to_datetime(timeDf['publishedAt'])
```

    <ipython-input-134-8242d260764d>:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      timeDf['NewDateTime'] = pd.to_datetime(timeDf['publishedAt'])
    


```python
# Convert datetime object to week, month and day
timeDf['week'] = timeDf['NewDateTime'].dt.week
timeDf['day'] = timeDf['NewDateTime'].dt.day
timeDf['month'] = timeDf['NewDateTime'].dt.month
```

    <ipython-input-135-24104e6ccdd9>:2: FutureWarning: Series.dt.weekofyear and Series.dt.week have been deprecated. Please use Series.dt.isocalendar().week instead.
      timeDf['week'] = timeDf['NewDateTime'].dt.week
    <ipython-input-135-24104e6ccdd9>:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      timeDf['week'] = timeDf['NewDateTime'].dt.week
    <ipython-input-135-24104e6ccdd9>:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      timeDf['day'] = timeDf['NewDateTime'].dt.day
    <ipython-input-135-24104e6ccdd9>:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      timeDf['month'] = timeDf['NewDateTime'].dt.month
    


```python
# The protests started on Jan 18, 2023
# https://indianexpress.com/article/sports/sport-others/wrestlers-protest-timeline-from-jantar-mantar-sit-in-to-nearly-immersing-medals-in-ganga-8649894/
```


```python
word_counts_beforeRegex_by_week = {}
```


```python
%%time
for index, row in timeDf.iterrows():
    text = row['textOriginal']
    week = row['week']
    words = text.split()
    slur_word_count = sum(1 for word in words if word in slur_words)
    word_counts_beforeRegex_by_week[week] = word_counts_beforeRegex_by_week.get(week, 0) + slur_word_count
```

    CPU times: user 3min 7s, sys: 750 ms, total: 3min 8s
    Wall time: 3min 9s
    


```python
# word_counts_beforeRegex_by_week
sorted_word_counts_beforeRegex_by_week = dict(sorted(word_counts_beforeRegex_by_week.items()))
```


```python
sum(word_counts_beforeRegex_by_week.values())
```




    22989




```python
# sorted_word_counts_beforeRegex_by_week
```


```python
# pattern = r'(?i)(' + '|'.join(slur_words_set) + r')'
pattern = r"(?i)\b(" + '|'.join(slur_words) + r")\b"
```


```python
compiled_pattern = re.compile(pattern, re.IGNORECASE)
```


```python
%%time
# counts2 = timeDf.groupby('week')['textOriginal'].apply(lambda x: x.str.contains(pattern, case=False).sum())
word_counts_afterRegex_by_week = timeDf.groupby('week')['textOriginal'].apply(lambda x: x.str.contains(compiled_pattern).sum())
```

    <timed exec>:2: UserWarning: This pattern is interpreted as a regular expression, and has match groups. To actually get the groups, use str.extract.
    

    CPU times: user 4min 2s, sys: 441 ms, total: 4min 2s
    Wall time: 4min 4s
    


```python
# word_counts_afterRegex_by_week
```


```python
word_counts_afterRegex_by_week.sum()
```




    34645




```python
'''
Lets plot the frequceny of slur words used vs time (week)
The week number denoted in the plot is the number of the week in the year (out of 52)
Week 17 - Apr. 24, 2023 - a day after protests started outside jantar mantar
Week 22 - May 29, 2023 -  Delhi Police detains top wrestlers
          May 30, 2023 - The wrestlers, hurt by the way the police had detained them and then filed FIRs against them,
                            announce they will go to Har Ki Pauri in Haridwar to 'immerse their medals' in the River Ganga.
Week 28 - July 10, 2023 - last day till I scrapped the data
'''
```




    "\nLets plot the frequceny of slur words used vs time (week)   \nThe week number denoted in the plot is the number of the week in the year (out of 52)\nWeek 17 - Apr. 24, 2023 - a day after protests started outside jantar mantar\nWeek 22 - May 29, 2023 -  Delhi Police detains top wrestlers\n          May 30, 2023 - The wrestlers, hurt by the way the police had detained them and then filed FIRs against them,\n                            announce they will go to Har Ki Pauri in Haridwar to 'immerse their medals' in the River Ganga.\nWeek 28 - July 10, 2023 - last day till I scrapped the data\n"




```python
keys_before_regex = list(sorted_word_counts_beforeRegex_by_week.keys())[8:]
values_before_regex = list(sorted_word_counts_beforeRegex_by_week.values())[8:]

keys = word_counts_afterRegex_by_week.index[8:]
values = word_counts_afterRegex_by_week.values[8:]

plt.plot(keys_before_regex, values_before_regex, color='blue', label='before regex')
plt.plot(keys, values, color='red', label='after regex')

plt.xlabel('Week')
plt.ylabel('Count of Slur Words')
plt.title('Count of Slur Words per week')
plt.xticks(keys)
plt.legend(loc='best', fontsize='small')
plt.show()
```


    
![png](youtube_comments_analysis_files/youtube_comments_analysis_93_0.png)
    



```python
keys_before_regex = list(sorted_word_counts_beforeRegex_by_week.keys())[8:]
values_before_regex = list(sorted_word_counts_beforeRegex_by_week.values())[8:]

keys = word_counts_afterRegex_by_week.index[8:]
values = word_counts_afterRegex_by_week.values[8:]

plt.semilogy(keys_before_regex, values_before_regex, color='blue', label='before regex')
plt.semilogy(keys, values, color='red', label='after regex')

plt.xlabel('Week')
plt.ylabel('Count of Slur Words')
plt.title('Count of Slur Words per week')
plt.xticks(keys)
plt.legend(loc='best', fontsize='small')
plt.show()
```


    
![png](youtube_comments_analysis_files/youtube_comments_analysis_94_0.png)
    



```python
# the pattern pretty much remains the same, only the count of slur words are increasing
```


```python
# Lets now look at the like counts accross the timeline of these hateful tweets.
```


```python
word_likes_per_week = {}
top_sentences_per_week = []
```


```python
timeDf['likeCount'] = timeDf['likeCount'].astype(int)
```

    <ipython-input-153-2b29a1900db1>:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      timeDf['likeCount'] = timeDf['likeCount'].astype(int)
    


```python
pattern = r"(?i)\b(" + '|'.join(slur_words) + r")\b"
compiled_pattern = re.compile(pattern, re.IGNORECASE)
```


```python
%%time
for index, row in timeDf.iterrows():
    text = row['textOriginal']
    week = row['week']
    like_count = row['likeCount']
    matches = compiled_pattern.findall(text)
    if matches:
        top_sentences_per_week.append([text, matches, like_count, week])
    for word in matches:
        if week not in word_likes_per_week:
            word_likes_per_week[week] = 0
        word_likes_per_week[week] += like_count
```

    CPU times: user 5min 15s, sys: 997 ms, total: 5min 16s
    Wall time: 5min 18s
    


```python
# word_likes_per_week
```


```python
# top_sentences_per_week
```


```python
x = sorted(word_likes_per_week.keys())
y = [word_likes_per_week[week] for week in x]
```


```python
plt.plot(x[1:], y[1:])
plt.xlabel('Week')
plt.ylabel('Total Like Count')
plt.title('Total Like Count per Week for Identified Words')
plt.xticks(x[1:])
plt.show()
```


    
![png](youtube_comments_analysis_files/youtube_comments_analysis_104_0.png)
    



```python
plt.semilogy(x[1:], y[1:])
plt.xlabel('Week')
plt.ylabel('Total Like Count')
plt.title('Total Like Count per Week for Identified Words - Semilogy Scale')
plt.xticks(x[1:])
plt.show()
```


    
![png](youtube_comments_analysis_files/youtube_comments_analysis_105_0.png)
    



```python
# These plots could be showing these results because it could be also from the bias in dataset sampling.
```


```python
def detect_slur(text):
    if compiled_pattern.search(text):
        return 'slur'
    else:
        return 'not_slur'
```


```python
%%time
timeDf['detect'] = timeDf['textOriginal'].apply(detect_slur)
```

    CPU times: user 3min 59s, sys: 428 ms, total: 3min 59s
    Wall time: 4min 1s
    

    <timed exec>:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    


```python
slurDf = timeDf[timeDf['detect'] == 'slur']
```


```python
# result_df = slurDf.loc[slurDf.groupby('week')['likeCount'].idxmax()]
result_df = slurDf.groupby('week').apply(lambda group: group.nlargest(3, 'likeCount')).reset_index(drop=True)
```


```python
result_dict = {}
```


```python
for index, row in result_df.iterrows():
    week_number = row['week']
    text = row['textOriginal']
    like_count = row['likeCount']

    if week_number not in result_dict:
        result_dict[week_number] = []

    result_dict[week_number].append({'text': text, 'likeCount': like_count})
```


```python
# result_dict
```


```python
'''
check if these hateful tweets are replies or not too
'''
```




    '\ncheck if these hateful tweets are replies or not too\n'




```python
slur_counts_replies_by_week = {}
slur_counts_non_replies_by_week = {}
```


```python
%%time
for index, row in timeDf.iterrows():
    text = row['textOriginal']
    matches = re.findall(r"(?i)\b(" + '|'.join(slur_words) + r")\b", text)
    parent_id = row['parentId']
    week_identifier = row['week']
    if matches:
        if pd.notna(parent_id):
            if week_identifier not in slur_counts_replies_by_week:
                slur_counts_replies_by_week[week_identifier] = 0
            slur_counts_replies_by_week[week_identifier] += 1
        else:
            if week_identifier not in slur_counts_non_replies_by_week:
                slur_counts_non_replies_by_week[week_identifier] = 0
            slur_counts_non_replies_by_week[week_identifier] += 1
```

    CPU times: user 6min 5s, sys: 1.04 s, total: 6min 6s
    Wall time: 6min 8s
    


```python
sorted_slur_counts_replies_by_week = {k: v for k, v in sorted(slur_counts_replies_by_week.items(), key=lambda item: item[0])}
sorted_slur_counts_non_replies_by_week = {k: v for k, v in sorted(slur_counts_non_replies_by_week.items(), key=lambda item: item[0])}
```


```python
weeks = list(sorted_slur_counts_replies_by_week.keys())[1:]
replies_counts = list(sorted_slur_counts_replies_by_week.values())[1:]
non_replies_counts = list(sorted_slur_counts_non_replies_by_week.values())[1:]
# plt.figure(figsize=(10, 6))
plt.plot(weeks, replies_counts, label='Replies', color='r')
plt.plot(weeks, non_replies_counts, label='Comment', color='b')
plt.xlabel('Week')
plt.ylabel('Count')
plt.title('Replies or Comments of Slur Text by Week')
plt.xticks(weeks)
plt.legend()
plt.tight_layout()
plt.show()
```


    
![png](youtube_comments_analysis_files/youtube_comments_analysis_118_0.png)
    

