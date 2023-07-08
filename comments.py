import csv
from datetime import datetime as dt

comments = []
# today = dt.now().strftime('%H:%M:%S')
# today = dt.today().strftime('%d-%m-%Y')
today = datetime.now().strftime("%Y-%m-%d")
# print(today)

def process_comments(response_items, csv_output=False):

    for res in response_items:

        # loop through the replies
        if 'replies' in res.keys():
            for reply in res['replies']['comments']:
                comment = reply['snippet']
                comment['commentId'] = reply['id']
                comments.append(comment)
        else:
            comment = {}
            comment['snippet'] = res['snippet']['topLevelComment']['snippet']
            comment['snippet']['parentId'] = None
            comment['snippet']['commentId'] = res['snippet']['topLevelComment']['id']

            comments.append(comment['snippet'])

    if csv_output:
         make_csv(comments)
    
    print(f'Finished processing {len(comments)} comments.')
    return comments

def make_csv(comments, channelID=None):
    header = comments[0].keys()

    if channelID:
#         filename = f'comments_{channelID}_{today}.csv'
        filename = f'comments.csv'
    else:
#         filename = f'comments_{today}.csv'
        filename = f'comments.csv'

    with open(filename, 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(comments)

