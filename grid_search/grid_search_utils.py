import os
import re
import pandas as pd
    
def get_grid_search_results(directory, len_lines=125):

    # results should be stored in a pandas dataframe and exported as csv
    data = pd.DataFrame()

    # go through all files in grid search results folder
    directory = directory
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".out"):
            with open(os.path.join(directory, filename), "r") as file:
                lines = file.readlines()
                
                if len(lines) == len_lines: # if files have less lines, the search did not run for 100 epochs
                    header = lines[0].strip()
                    if header == "ndevices 0":
                        header = lines[1].strip()

                    # create a dictionary with parameters
                    parameters = {}
                    matches = re.findall(r"--(.+?)=(.+?)\"", header)
                    for match in matches:
                        parameters[match[0]] = match[1]

                    # get results
                    final_train = lines[-3].strip()
                    final_test = lines[-2].strip()

                    train_loss, train_acc = re.findall(r"\"loss\": (.+?), \"acc\": (.+?),", final_train)[0]
                    test_loss, test_acc = re.findall(r"\"loss\": (.+?), \"acc\": (.+?),", final_test)[0]

                    # update the parameters dictionary with train and test accuracies
                    parameters['train_loss'] = train_loss
                    parameters['train_accuracy'] = train_acc
                    parameters['test_loss'] = test_loss
                    parameters['test_accuracy'] = test_acc

                    df = pd.DataFrame(parameters, index=[0])

                    data = pd.concat([data, df], ignore_index=True)

    # sort and save as csv
    data = data.astype(float)

    data[['attributes', 'values', 'game_size', 'batch_size', 'speaker_hidden_size']] = data[['attributes', 'values', 'game_size', 'batch_size', 'speaker_hidden_size']].astype(int)
    data = data.sort_values(by=['attributes', 'values', 'game_size', 'batch_size', 'learning_rate', 'speaker_hidden_size', 'temperature', 'temp_update'])

    data.to_csv('results_' + directory + '.csv', index=False)
    print("Saved data as " + 'results_' + directory + '.csv')
