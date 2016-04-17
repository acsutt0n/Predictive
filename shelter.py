# shelter stuff

"""
This is based on some excellent work from juandoso,
   gives a more detailed (but still low effort) random forest.
I wanted to avoid subjective assessments of animal size or
   perceived aggressiveness of breed which might be great predictors
   but make analysis too murky -- unless a workaround can be found.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt



#########################################################################
# Helper functions

def convert_date(dat):
  # Categorical conversion for month, day, year
  d = dt.datetime.strptime(dat, "%Y-%m-%d %H:%M:%S")
  return d.year, d.month, d.isoweekday()



def gender(colin):
  # Return intact (1), not intact (2) or unknown (2)
  sex, intact, projsex = [], [], []
  try: colin = colin.values
  except: pass
  for u in colin:
    try:
      if 'spay' in u.lower() or 'neut' in u.lower():
        intact.append(0.)
      elif 'intact' in u.lower():
        intact.append(1.)
      else:
        intact.append(2.)
    except:
      intact.append(2.)
    try:
      if 'male' in u.lower():
        sex.append(1.)
        projsex.append(1.)
      elif 'female' in u.lower():
        sex.append(0.)
        projsex.append(0.)
      else:
        sex.append(2.)
        projsex.append(0.5)
    except:
      sex.append(2.)
      projsex.append(0.5)
  return sex, projsex, intact



def get_age(agecol, fillwmean=True):
  # Return the age in days
  age = []
  try: agecol = agecol.values
  except: pass
  for d in agecol:
    try:
      dstring = str(d)
    except:
      dstring = '1 day'
    dsplit = dstring.split(' ')
    try:
      num = int(dsplit[0])
      if 'day' in dsplit[1].lower():
        age.append(num*1.)
      elif 'week' in dsplit[1].lower():
        age.append(num*7.)
      elif 'month' in dsplit[1].lower():
        age.append(num*30.)
      elif 'year' in dsplit[1].lower():
        age.append(num*365.)
      else:
        age.append(np.nan)
    except:
      age.append(np.nan)
  mage = pd.Series(age).mean()
  age = [mage if pd.isnull(h) else h for h in age]
  return age



def hours(datetime):
  # What hour of the day was the animal brought in?
  dates = []
  try: datetime = datetime.values
  except: pass
  for d in datetime:
    dsplit = d.split(' ')[1].split(':')[0]
    dates.append(float(dsplit))
  return dates



def efficient_dummies(df, columns, keep_nan=False):
  """
  pd.get_dummies is very inefficient, creating so many columns.
  Instead, this approach just assigns ints based on values for that column,
  a.k.a it "floats" strings.
  Do this with the combined df to ensure train and test have same int vals.
  This *may* be slower, but it will use less memory.
  """
  for col in columns:
    colset = list(range(len(set(df[col]))))
    nan = [1 if pd.isnull(c) else 0 for c in colset]
    if sum(nan) > 1: # Multiple nan's registerred as separate values
      colset = list(range(len(colset)-sum(nan)+1)) # Only one nan kept
    newcol = [colset.index(j) for j in df[col]]
  




# Also get Mix(0,1), Dog(0,1), Named(0,1) 

# Also need to code the Outcome as a number:
to_num = {'Adoption': 0, 'Died': 1, 'Euthanasia': 2, 
          'Return_to_owner': 3, 'Transfer': 4}
from_num = {0: 'Adoption', 1: 'Died', 2: 'Euthanasia', 
            3: 'Return_to_owner', 4: 'Transfer'}
# This is the expected order of the output as well, so a value of 0
#   allows us to put a 1 in col 0 and a 0 elsewhere



################################################################
# Scripting stuff


def prepare_df(df, is_test=False):
  """
  Remove and add a bunch of stuff; if test is True then
  Returns DF, ID and targets (unless test=True, then just DF and ID)
  """
  # Named
  named = [1 if not pd.isnull(h) else 0 for h in df.Name.values]
  df['Named'] = named
  df.drop('Name', axis=1, inplace=True)
  # Date and time stuff
  df['Year'], df['Month'], df['DayOfWeek'] = \
    zip(*df['DateTime'].map(convert_date))
  df['Hours'] = hours(df['DateTime'].values)
  df.drop('DateTime', axis=1, inplace=True)
  # Do ID stuff
  if is_test:
    ids = df[['ID']]
  else:
    ids = df[['AnimalID']]
  return df, ids



def shelter_data(test, train):
  """
  Remove and add a bunch of stuff; if test is True then
  Returns DF, ID and targets (unless test=True, then just DF and ID)
  """
  # Prepare dfs
  train, train_ids = prepare_df(train, is_test=False)
  test, test_ids = prepare_df(test, is_test=True)
  # Get targets
  train_outcome = train['OutcomeType']
  
  # Get dummy indicator variables; best way is to combine dfs first
  train['train'] = 1
  test['train'] = 0
  combined = pd.concat([train, test])
  combined_dummy = pd.get_dummies(combined, columns=combined.columns)
  # Separate back out
  train = combined_dummy[combined_dummy['train_1'] == 1]
  test = combined_dummy[combined_dummy['train_0'] == 1]
  # And drop the separation columns
  train.drop(['train_0', 'train_1'], axis=1, inplace=True)
  test.drop(['train_0', 'train_1'], axis=1, inplace=True)
  
  return test, test_ids, train, train_ids, train_outcome



def shelter_forest(train, train_result, train_ids, test, test_ids):
  """
  This is run after prepare_df cleans the data and shelter_data produces
  the dummy variables. 
  """
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.cross_validation import train_test_split
  
  # Train the RF on some CV data
  x_train, x_val, y_train, y_val = train_test_split(train, 
                                                    train_result,
                                                    test_size=0.1) #10% of train as test
  forest = RandomForestClassifier(n_estimators=250, n_jobs=2)
  forest.fit(x_train, y_train)
  y_pred_val = forest.predict(x_val)
  # Test
  from skelearn.metrics import classification_report, accuracy_score
  print(classification_report(y_val, y_pred_val))
  print(accuracy_score(y_val, y_pred_val))
  
  # On the complete training set
  forest = RandomForestClassifier(n_estimators=500, n_jobs=2)
  forest.fit(train, train_result)
  y_pred = forest.predict_proba(test)
  return y_pred



def run_shelter(test_file, train_file, output_template, output_file):
  """
  Run everything except analysis and plotting.
  """
  # Prepare the data
  test_csv, train_csv = pd.read_csv(test_file), pd.read_csv(train_file)
  test, test_ids, train, train_ids, train_outcome =\
    shelter_data(test_csv, train_csv)
  
  # Run the forest
  y_pred = shelter_forest(train, train_outcome, train_ids, test, test_ids)
  
  # Load in results
  output = pd.read_csv(output_template)
  collist = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
  for c in range(len(collist)):
    output[collist[c]] = y_pred[:,c]
  output.to_csv(output_file, index=False)
  print('Output written to %s' %output_file)
  return
  



### Notes
"""
# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(ytrain, xtrain)

# Take the same decision trees and run it on the test data
ytest = forest.predict(xtest)
"""


#########################################################################
# Analysis







#########################################################################

# Output as simple csv

def create_output(xtestdf, ytest_, nam='output6.csv'):
  # Need xtestdf for the IDs
  res = np.zeros((len(ytest_),5))
  for y in range(len(ytest_)):
    res[y, ytest_[y]] = 1
  #
  collist = ['ID', 'Adoption', 'Died', 'Euthanasia', 
             'Return_to_owner', 'Transfer']
  output = pd.DataFrame(columns=collist)
  output.ID = list(xtestdf.ID.values)
  for colnam in collist[1:]:
    output[colnam] = res[:, collist.index(colnam)-1]
  output.to_csv(nam, index=False)
  print('Saved output file as %s' %nam)
  return













