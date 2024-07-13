"""this is an example script for preparing the model input and postprocessing the model outputs"""


# from thefuzz import fuzz, process
from rapidfuzz import process, fuzz
import pandas as pd
from io import StringIO

prompt_config = {
    'system': """You are a helpful assistant. You are an experienced clinician and you are familiar with writing and understanding clinical notes.""",
    'background':
"""A clinical note contains multiple sections like family history, allergies and history of present illness.

Given a clinical note as an input, please separate the note into sections and output their section names. Also specify where the section starts and ends. Use section names from one of the following:

{description}

For example, the output format should be 
Section 1: "Social history"
Starts at: "SOCIAL HISTORY:"
Ends at: "no alcohol."

Section 2: "Family history"
Starts at: "FAMILY HISTORY:"
Ends at: "parents do not have diabetes."
""",
    'example_config': {
        'partial_example_template':
"""Input:

{note}

Output:""",
        'full_example_template':
"""Input:

{note}

Output:
{output}"""
},
    'prompt_template':
"""{background}

{example}"""
}


def make_messages(train_data_df, val_data_df, test_data_df, note_df, description_df, prompt_config, num_example=0):
    """Make chatgpt input """
    ## make example
    example_list = []
    for i in range(num_example):
        full_example = make_messages_full_example(train_data_df=train_data_df, val_data_df=val_data_df, note_df=note_df, description_df=description_df, prompt_config=prompt_config)
        example_list.append(full_example)
    partial_example = make_messages_partial_example(note_df, prompt_config)
    example_list.append(partial_example)
    example = '\n\n'.join(example_list)

    ## make background
    description = make_messages_description(description_df)
    background = prompt_config['prompt_config']['background'].replace('{description}', description)

    ## put together prompt
    prompt = prompt_config['prompt_config']['prompt_template'].replace('{background}', background).replace('{example}', example)
    messages = [{"role": "system", "content": prompt_config['prompt_config']['system']}, {"role": "user", "content": prompt}]

    return messages

def make_messages_full_example(train_data_df, val_data_df, note_df, description_df, prompt_config):
    train_val_data_df = pd.concat([train_data_df, val_data_df])
    note_id = train_val_data_df['note_id'].sample(1).item()
    note_df = train_val_data_df[train_val_data_df['note_id'] == note_id]
    note = note_df['note_text'].iloc[0]
    output = create_output(note_df=note_df, description_df=description_df)
    full_example = prompt_config['prompt_config']['example_config']['full_example_template'].replace('{note}', note).replace('{output}', output)
    return full_example

def create_output(note_df, description_df, word_size=10):
    one_section_output_list = []
    for index, section in note_df.iterrows():  # this index is not starting from 0 because note_df is a subset of a larger df
        one_section_output = '''Section {}: "{}"
Starts at: "{}"
Ends at: "{}"'''.format(str(section['section_index'] + 1),
                        section['section_name'],
                        ' '.join(section['section_text'].split(' ')[:word_size]).strip(),
                        ' '.join(section['section_text'].split(' ')[-word_size:]).strip())
        one_section_output_list.append(one_section_output)
    output = '\n\n'.join(one_section_output_list)
    return output

def make_messages_partial_example(note_df, prompt_config):
    note = note_df['note_text'].iloc[0]
    partial_example = prompt_config['prompt_config']['example_config']['partial_example_template'].replace('{note}', note)
    # print(partial_example)
    return partial_example

def make_messages_description(description_df):
    category_and_description_string_list = []
    for category_name in description_df['category_name'].unique():
        category_df = description_df[description_df['category_name'] == category_name][['category_name', 'description']]
        assert category_df.drop_duplicates().shape[0] == 1
        description = category_df['description'].iloc[0]
        category_and_description_string_list.append(f'{category_name}: {description}')
    category_description = '\n'.join(category_and_description_string_list)
    return category_description

def is_good_output(output):
    """Check if chatgpt output is in the expected format"""
    chunk_start_index, chunk_end_index = None, None
    for chunk_index, chunk_text in enumerate(output.split('\n')):
        if chunk_text.strip().startswith('Section 1:'):
            chunk_start_index = chunk_index
        if chunk_text.strip().startswith('Ends at: '):
            chunk_end_index = chunk_index + 1

    if chunk_start_index is None or chunk_end_index is None:
        return False

    useful_output = '\n'.join(output.split('\n')[chunk_start_index: chunk_end_index])

    good_section_info_list = []
    for section_index, section_info in enumerate(useful_output.split('\n\n')):
        section_info = section_info.strip()
        line_series = pd.Series(section_info.split('\n'))

        # good_section_info = line_series.iloc[0].endswith('"') and line_series.str.startswith(f'Section {section_index + 1}: "').sum() == 1 and line_series.str.startswith('Starts at: "').sum() == 1 and line_series.str.startswith('Ends at: "').sum() == 1
        good_section_info = line_series.str.startswith(f'Section {section_index + 1}: ').sum() == 1 and line_series.str.startswith('Starts at: "').sum() == 1 and line_series.str.startswith('Ends at: "').sum() == 1

        good_section_info_list.append(good_section_info)
    good_output = all(good_section_info_list)
    return good_output  # good_output = False

def format_output(note_df, output, description_df, fuzzy_name=False, fuzzy_sentence=False):
    """Format chatgpt output into a dataframe """
    processing_log = {}
    note_text = note_df['note_text'].iloc[0]
    ## 1. get the good part from output
    chunk_start_index, chunk_end_index = None, None
    for chunk_index, chunk_text in enumerate(output.split('\n')):
        if chunk_text.strip().startswith('Section 1: '):  # the beginning of the valid output
            chunk_start_index = chunk_index
        if chunk_text.strip().startswith('Ends at: "'):  # the end of the valid output
            chunk_end_index = chunk_index + 1

    output = '\n'.join(output.split('\n')[chunk_start_index: chunk_end_index]).strip()
    if not output.endswith('"'):  # the last section, the end text is multi line, cannot get lines other than the first line, in the case, need to add " to enclose it
        # print('ends with " ')
        # print(repr(output))
        output += '"'  # Ends at: "asdfasdf
    if output.endswith('Ends at: "'):
        # print('ends with ends at: "')
        # print(repr(output))
        output += '"'  # Ends at: "

    ## 2. structurize output
    section_info_df_row_list = []
    for section_index, section_name_start_text_end_text in enumerate(output.split('\n\n')):
        section_name_start_text_end_text = section_name_start_text_end_text.strip()

        section_name = section_name_start_text_end_text.split('\nStarts at: ')[0].split(': ')[1].strip()
        if section_name.startswith('"'):
            section_name = section_name[1:]
        if section_name.endswith('"'):
            section_name = section_name[:-1]
        if fuzzy_name:
            if description_df['category_name'][description_df['category_name'].str.lower() == section_name.lower()].shape[0] == 1:  # TODO technique 1: case insensitive section name match
                section_name = description_df['category_name'][description_df['category_name'].str.lower() == section_name.lower()].iloc[0]
        section_start_text = section_name_start_text_end_text.split('\nStarts at: "')[1].split('"\nEnds at: "')[0]
        section_end_text = section_name_start_text_end_text.split('\nStarts at: "')[1].split('"\nEnds at: "')[1][:-1]

        if fuzzy_sentence:
            if section_start_text.endswith(':') or section_start_text.endswith('.'):  # TODO technique 2: remove ':' or '.'
                section_start_text = section_start_text[:-1]
        if section_end_text.endswith(':') or section_end_text.endswith('.'):
            section_end_text = section_end_text[:-1]

        section_info_df_row = {'section_name': section_name, 'section_start_text': section_start_text, 'section_end_text': section_end_text}
        section_info_df_row_list.append(section_info_df_row)
    section_info_df = pd.DataFrame(section_info_df_row_list)

    ## 3. filter invalid section names
    size_before_section_name_filter = section_info_df.shape[0]
    invalid_section_name_list = section_info_df[section_info_df['section_name'].apply(lambda section_name: section_name not in description_df['category_name'].to_list())]['section_name'].to_list()
    invalid_section_name_index_list = section_info_df[section_info_df['section_name'].apply(lambda section_name: section_name not in description_df['category_name'].to_list())].index.to_list()
    section_info_df = section_info_df[section_info_df['section_name'].apply(lambda section_name: section_name in description_df['category_name'].to_list())]
    print(f"(ahead) invalid section start text {(section_info_df['section_start_text'].apply(lambda section_start_text: note_df['note_text'].iloc[0].find(section_start_text)) == -1).sum()}/{size_before_section_name_filter}")
    print(f"(ahead) invalid section end text {(section_info_df['section_end_text'].apply(lambda section_end_text: note_df['note_text'].iloc[0].find(section_end_text)) == -1).sum()}/{size_before_section_name_filter}")
    print(f'invalid section names {size_before_section_name_filter - section_info_df.shape[0]}/{size_before_section_name_filter}')
    print(invalid_section_name_list, invalid_section_name_index_list)

    processing_log['num_section'] = size_before_section_name_filter
    processing_log['num_valid_section_start'] = size_before_section_name_filter - (section_info_df['section_start_text'].apply(lambda section_start_text: -1 if section_start_text == '' else note_df['note_text'].iloc[0].find(section_start_text)) == -1).sum()
    processing_log['num_valid_section_end'] = size_before_section_name_filter - (section_info_df['section_end_text'].apply(lambda section_end_text: -1 if section_end_text == '' else note_df['note_text'].iloc[0].find(section_end_text)) == -1).sum()
    processing_log['num_valid_section_name'] = size_before_section_name_filter - (size_before_section_name_filter - section_info_df.shape[0])

    def fuzzy_find_best_matched_sequence(string, note_text):
        if string in note_text:
            print(f"Most similar sequence (score=100): '{string}' <- '{string}'")
            return string

        all_sequences = [note_text[i:j] for i in range(len(note_text)) for j in range(i + 1, len(note_text) + 1)]
        best_match = process.extractOne(string, all_sequences, scorer=fuzz.ratio)
        most_similar_sequence = best_match[0]
        similarity_score = best_match[1]

        print(f"Most similar sequence (score={str(similarity_score)}): '{most_similar_sequence}' <- '{string}'")
        if similarity_score < 90:
            return string
        else:
            return most_similar_sequence

    ## 3.1 TODO technique 2: fussy match start text
    if fuzzy_sentence:
        section_info_df['section_start_text'] = section_info_df['section_start_text'].apply(lambda section_start_text: fuzzy_find_best_matched_sequence(section_start_text, note_text))

    ## 4. assign section start_char index
    section_info_df['section_start_char_index'] = section_info_df['section_start_text'].apply(lambda section_start_text: -1 if section_start_text.strip() == '' else note_text.find(section_start_text))  # sometimes the start text is literally '' -> empty string, but can str.find gives this a section_start_text of 0. Force it to -1

    ## 5. filter invalid section start char index
    size_before_start_char_filter = section_info_df.shape[0]
    section_info_df = section_info_df[~(section_info_df['section_start_char_index'] == -1)]
    print(f'invalid section start text: {size_before_start_char_filter - section_info_df.shape[0]}/{size_before_start_char_filter}')

    ## 5.1 TODO technique 2: fussy match end text
    if fuzzy_sentence:
        section_info_df['section_end_text'] = section_info_df['section_end_text'].apply(lambda section_end_text: fuzzy_find_best_matched_sequence(section_end_text, note_text))

    ## 6. sort by section start char index
    size_before_duplicated_start_char_filter = section_info_df.shape[0]
    section_info_df = section_info_df.drop_duplicates(subset=['section_start_char_index'])  # by default keeps the first occurrence, need to do this for the bad performance of the bad models
    section_info_df = section_info_df.sort_values('section_start_char_index')
    print(f'duplicated section start char: {size_before_duplicated_start_char_filter - section_info_df.shape[0]}/{size_before_duplicated_start_char_filter}')


    ## 7. create the column of next section's start
    section_info_df.insert(0, 'section_index', range(section_info_df.shape[0]))
    section_info_df['next_section_start_char_index'] = section_info_df['section_start_char_index'].shift(-1).fillna(len(note_text)).apply(int)

    ## 8. assign section end char index
    def find_section_end_char_index(row, note_text):
        # sometimes the end text is also literally '' -> empty string, but can str.find gives this row['section_start_char_index']. Force it to -1
        initial_end_char_index = -1 if row['section_end_text'] == '' else note_text.find(row['section_end_text'], row['section_start_char_index'], row['next_section_start_char_index'])
        if initial_end_char_index == -1:
            final_end_char_index = row['next_section_start_char_index']
        else:
            final_end_char_index = initial_end_char_index + len(row['section_end_text'])
        row['section_end_char_index'] = final_end_char_index
        return row

    section_info_df = section_info_df.apply(find_section_end_char_index, axis=1, note_text=note_text)

    ## 9. slice section text
    def slice_section_text(row, note_text):
        row['section_text'] = note_text[row['section_start_char_index']: row['section_end_char_index']]
        return row
    section_info_df = section_info_df.apply(slice_section_text, axis=1, note_text=note_text)

    ## 10. add note id
    section_info_df.insert(0, 'note_id', note_df['note_id'].iloc[0])
    # for token_type, token_count in usage_dict.items():
    #     section_info_df[token_type] = token_count

    if section_info_df.shape[0] == 0:  # sometimes it is an empty df, if returned and concated with other df, it changes the int column to float, so return None which will be ignored in pd.concat
        section_info_df = None
    return section_info_df, processing_log

def get_output():  #  from mtsamples
    desciprion = '''
Section 1: "Admission diagnosis"
Starts at: "ADMITTING DIAGNOSIS"
Ends at: "left foot."

Section 2: "Patient procedures"
Starts at: "PROCEDURES:"
Ends at: "left foot with culture and sensitivity."

Section 3: "Social history"
Starts at: "DISCHARGE DIAGNOSIS"
Ends at: "left foot."

        '''
    return desciprion

def get_description():
    desciprion = '''
    
section_name,category_name,description
"Unknown","Unknown","the given information is insufficient for inferring the section's section name"
"Subsection","Unknown","the given information is insufficient for inferring the section's section name"
"Admit Date","Admit date","the date of the patient's admit, example section headings include 'admission date'"
"Discharge Date","Discharge date","the date of the patient's discharge, example section headings include 'discharge date'"
"Reason for Admission","Reason for admission","the reason why the patient was admitted to the hospital, example section headings include 'chief complaint', 'history of present illness', 'reason for hospitalization'"
"Discharge Instructions","Discharge instructions","guidelines and information provided to a patient when they are being discharged from the hospital, example section headings include 'todo', 'plan', 'diet', 'discharge instructions'"
"Discharge Diagnoses","Discharge diagnosis","diagnosis assigned to a patient at the time of their discharge from the hospital, example section headings include 'principle discharge diagnosis', 'discharge diagnosis'"
"Discharge Medications","Discharge medications","medications prescribed to a patient at the time of their discharge from the hospital, example section headings include 'discharge medications', 'medications on discharge'"
"Admission Diagnoses","Admission diagnosis","medical condition or diagnosis that leads to a patient's admission to a healthcare facility, example section headings include 'admit diagnosis', 'principle diagnosis'"
"Procedures","Patient procedures","medical interventions, treatments, or actions performed by healthcare professionals to diagnose, treat, manage, or prevent various medical conditions, example section headings include 'operations and procedures', 'procedures', 'principle procedures', 'major surgical or invasive procedures', 'special procedures and instructions'"
"Hospital Course","Hospital course","the sequence of events and medical interventions that occur during a patient's stay in a hospital, example section headings include 'hospital course', 'brief resume of hospital course'"
"Past Medical History","Past medical history","a patient's documented medical history of illnesses, conditions, surgeries, and significant medical events that have occurred prior to the current medical encounter, example section headings include 'past medical history'"
"Past Surgical History","Past surgical history","a patient's documented history of surgical procedures and operations they have undergone in the past, example section headings include 'past surgical history'"
"History","History of present illness","the patient's history of present illness, example section headings include 'history of present illness'"
"Physical","Physical examination","the patient's physical examination, example section headings include 'physical examination', 'review of systems'"
"Social History","Social history","a patient's background and lifestyle factors that can have a significant impact on their health and well-being, example section headings include 'social history'"
"Family History","Family history","a record of health-related information about an individual's immediate family members, such as parents, siblings, and children, example section headings include 'family history'"
"Allergies","Allergies","a dedicated portion of a patient's medical record or health profile that contains information about their known allergies, example section headings include 'allergies'"
"Followup","Follow-up","the follow-up information given to the patient after discharge, example section headings include 'follow up', 'follow up appointment', 'follow up services', 'instructions', 'dispositions', 'plan on discharge', 'patient dispositions'"
"Disposition","Follow-up","the follow-up information given to the patient after discharge, example section headings include 'follow up', 'follow up appointment', 'follow up services', 'instructions', 'dispositions', 'plan on discharge', 'patient dispositions'"
"Medications","Admission medications","substances, typically in the form of drugs or pharmaceuticals, that are used to diagnose, treat, manage, or prevent medical conditions, example section headings include 'medications', 'medications on admission'"
"Gynecologic History","Gynecologic history","a medical record or health profile pertains to a patient's reproductive and gynecological health"
"Other Diagnoses","Other diagnosis","diagnoses that are not discharge diagnosis or admission diagnosis, example section headings include 'associated diagnosis', 'other diagnosis',  'secondary diagnosis'"
"Service","Patient service","services done for the patient, example section headings include 'service'"
"Condition","Discharge condition","patient's condition on discharge, example section headings include 'discharge condition', 'condition on discharge'"
"Studies","Lab studies","laboratory studies for the patient, example section headings include 'laboratory data', 'laboratory studies', 'pertinent results'"
"Comments","Patient comments","comments left for patients, example section headings include 'additional comments', 'summary', 'addendum'"
"Admit Physician","Admitting physician","the physician who admits the patient"
"Attending","Attending physician","the patient's attending physician, example section headings include 'attending'"
    '''
    csv_file_like = StringIO(desciprion)
    df = pd.read_csv(csv_file_like)
    return df

def get_data():
      # a note from mtsamples https://mtsamples.com/site/pages/sample.asp?Type=89-Discharge%20Summary&Sample=1254-Abscess%20with%20Cellulitis%20-%20Discharge%20Summary
    dataset = '''
note_id,section_index,section_name,section_start_char_index,section_end_char_index,section_text,note_text
9423343AH,0,Admission Diagnosis,0,57,"ADMITTING DIAGNOSIS: Abscess with cellulitis, left foot.
","ADMITTING DIAGNOSIS: Abscess with cellulitis, left foot.

DISCHARGE DIAGNOSIS: Status post I&D, left foot.

PROCEDURES: Incision and drainage, first metatarsal head, left foot with culture and sensitivity.

HISTORY OF PRESENT ILLNESS: The patient presented to Dr. X's office on 06/14/07 complaining of a painful left foot. The patient had been treated conservatively in office for approximately 5 days, but symptoms progressed with the need of incision and drainage being decided."
9423343AH,1,Patient procedures,108,206,"PROCEDURES: Incision and drainage, first metatarsal head, left foot with culture and sensitivity.
","ADMITTING DIAGNOSIS: Abscess with cellulitis, left foot.

DISCHARGE DIAGNOSIS: Status post I&D, left foot.

PROCEDURES: Incision and drainage, first metatarsal head, left foot with culture and sensitivity.

HISTORY OF PRESENT ILLNESS: The patient presented to Dr. X's office on 06/14/07 complaining of a painful left foot. The patient had been treated conservatively in office for approximately 5 days, but symptoms progressed with the need of incision and drainage being decided."
        '''
    csv_file_like = StringIO(dataset)
    df = pd.read_csv(csv_file_like)
    return df


if __name__ == '__main__':
    pd.set_option('display.max_columns', 9)  # how many columns
    pd.set_option('display.max_colwidth', 30)  # show more for one column
    pd.set_option('display.max_rows', 100)  # show more for one column
    pd.set_option('display.expand_frame_repr', False)

    description_df = get_description()  # for discharge dataset
    data = get_data()  # using a public note from mtsample for demonstration
    new_prompt_config = {}
    new_prompt_config['prompt_config'] = prompt_config
    model_input = make_messages(train_data_df=data, val_data_df=data, test_data_df=data, note_df=data, description_df=description_df, prompt_config=new_prompt_config, num_example=0)
    model_input  # the input for vllm's chatgpt server
    model_output = get_output()  # two correct sections, one wrong section
    is_good = is_good_output(model_output)
    if is_good:
        processed_output, processing_log = format_output(data, model_output, description_df, fuzzy_name=False, fuzzy_sentence=False)
    else:
        # query again
        pass

