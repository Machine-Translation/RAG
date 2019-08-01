# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 20:00:10 2019

@author: Cory
"""


from music21 import converter, instrument, chord, stream, tempo, note
import numpy as np
import glob
import re
import os

def create_note(length, is_rest):
    """Creates a music21 note based on length and whether it is a rest or not"""
    if is_rest == True:
        new_note = note.Rest()
    else:
        new_note = note.Note()

    #new_note.offset = length
    new_note.duration.quarterLength = length
    new_note.storedInstrument = instrument.Piano()
    return new_note

def get_note_dic(int_to_note):
    """Builds a dictionary for all note types"""

    """
    These are lengths of notes in a measure of 4/4 time.
    EX: 4.0 = whole note
    EX: "2.5" = dotted half note
    EX: "1/3" = an eigth note in a triplet
    """
    lengths = ["4.0", "3.5", "3.0", "2.5", "2.0", "1.5", "1.0", "0.75", "2/3", "0.5", "1/3", "0.25", "1/12"]
    notes = []
    notes.append("|")
    for length in lengths:
        notes.append((length, True))
        notes.append((length, False))

    if int_to_note == True:
        return dict((id, n) for id, n in enumerate(notes))
    else:
        return dict((n, id) for id, n in enumerate(notes))

def get_prop_level(level):
    """Get the note proportions for a given level. A returned dictionar has is
    a pdf for given notes where the map is (id, probability)."""
    note_to_int = get_note_dic(False)
    if level == 1:
        prob_dic = dict()
        prob_dic[note_to_int.get(("1.0", True))] = 0.2  #quarter rest
        prob_dic[note_to_int.get(("1.0", False))] = "0.5" #quarter note
        prob_dic[note_to_int.get(("2.0", True))] = 0.1  #half rest
        prob_dic[note_to_int.get(("2.0", False))] = 0.2 #half note
        return prob_dic
    elif level == 2:
        prob_dic = dict()
        prob_dic[note_to_int.get(("0.5", True))] = 0.1  #eighth rest
        prob_dic[note_to_int.get(("0.5", False))] = 0.4 #eighth note

        prob_dic[note_to_int.get(("1.0", True))] = 0.1  #quarter rest
        prob_dic[note_to_int.get(("1.0", False))] = 0.2 #quarter note
        prob_dic[note_to_int.get(("2.0", True))] = 0.1  #half rest
        prob_dic[note_to_int.get(("2.0", False))] = 0.1 #half note
        return prob_dic
    elif level == 3:
        prob_dic = dict()
        prob_dic[note_to_int.get(("0.25", True))] = 0.05    #sixteenth rest
        prob_dic[note_to_int.get(("0.25", False))] = 0.05   #sixteenth note

        prob_dic[note_to_int.get(("0.5", True))] = 0.15     #eighth rest
        prob_dic[note_to_int.get(("0.5", False))] = 0.3     #eighth note
        prob_dic[note_to_int.get(("1.0", True))] = 0.15     #quarter rest
        prob_dic[note_to_int.get(("1.0", False))] = 0.2     #quarter note
        prob_dic[note_to_int.get(("2.0", True))] = 0.05     #half rest
        prob_dic[note_to_int.get(("2.0", False))] = 0.05    #half note
        return prob_dic
    elif level == 4:
        prob_dic = dict()
        prob_dic[note_to_int.get(("0.25", True))] = 0.2     #sixteenth rest
        prob_dic[note_to_int.get(("0.25", False))] = 0.4    #sixteenth note
        prob_dic[note_to_int.get(("0.5", True))] = 0.1      #eighth rest
        prob_dic[note_to_int.get(("0.5", False))] = 0.15    #eighth note
        prob_dic[note_to_int.get(("1.0", True))] = 0.05     #quarter rest
        prob_dic[note_to_int.get(("1.0", False))] = 0.05    #quarter note
        prob_dic[note_to_int.get(("2.0", True))] = 0.025    #half rest
        prob_dic[note_to_int.get(("2.0", False))] = 0.025   #half note
        return prob_dic
    elif level == 5:
        prob_dic = dict()
        prob_dic[note_to_int.get(("0.25", True))] = 0.2     #sixteenth rest
        prob_dic[note_to_int.get(("0.25", False))] = 0.3    #sixteenth note
        prob_dic[note_to_int.get(("0.5", True))] = 0.1      #eighth rest
        prob_dic[note_to_int.get(("0.5", False))] = 0.2     #eighth note
        prob_dic[note_to_int.get(("1.0", True))] = 0.05     #quarter rest
        prob_dic[note_to_int.get(("1.0", False))] = 0.1     #quarter note
        prob_dic[note_to_int.get(("2.0", True))] = 0.025    #half rest
        prob_dic[note_to_int.get(("2.0", False))] = 0.025   #half note
        return prob_dic
    elif level == 6:
        prob_dic = dict()
        prob_dic[note_to_int.get(("1.5", True))] = 0.05     #dotted quarter rest
        prob_dic[note_to_int.get(("1.5", False))] = "0.25"    #dotted quarter note

        prob_dic[note_to_int.get(("0.25", True))] = 0.05    #sixteenth rest
        prob_dic[note_to_int.get(("0.25", False))] = 0.2    #sixteenth note
        prob_dic[note_to_int.get(("0.5", True))] = 0.05     #eighth rest
        prob_dic[note_to_int.get(("0.5", False))] = 0.2     #eighth note
        prob_dic[note_to_int.get(("1.0", True))] = 0.05     #quarter rest
        prob_dic[note_to_int.get(("1.0", False))] = 0.1     #quarter note
        prob_dic[note_to_int.get(("2.0", True))] = 0.025    #half rest
        prob_dic[note_to_int.get(("2.0", False))] = 0.025   #half note
        return prob_dic
    elif level == 7:
        prob_dic = dict()
        prob_dic[note_to_int.get(("0.75", True))] = 0.05    #dotted eighth rest
        prob_dic[note_to_int.get(("0.75", False))] = "0.25"   #dotted eighth note

        prob_dic[note_to_int.get(("1.5", True))] = 0.075    #dotted quarter rest
        prob_dic[note_to_int.get(("1.5", False))] = 0.1     #dotted quarter note
        prob_dic[note_to_int.get(("0.25", True))] = 0.05    #sixteenth rest
        prob_dic[note_to_int.get(("0.25", False))] = 0.1    #sixteenth note
        prob_dic[note_to_int.get(("0.5", True))] = 0.05     #eighth rest
        prob_dic[note_to_int.get(("0.5", False))] = 0.1     #eighth note
        prob_dic[note_to_int.get(("1.0", True))] = 0.05     #quarter rest
        prob_dic[note_to_int.get(("1.0", False))] = 0.1     #quarter note
        prob_dic[note_to_int.get(("2.0", True))] = 0.025    #half rest
        prob_dic[note_to_int.get(("2.0", False))] = 0.05    #half note
        return prob_dic
    elif level == 8:
        prob_dic = dict()
        prob_dic[note_to_int.get(("1/3", True))] = 0.025    #triplet eighth rest
        prob_dic[note_to_int.get(("1/3", False))] = 0.2     #triplet eighth note

        prob_dic[note_to_int.get(("0.75", True))] = 0.05    #dotted eighth rest
        prob_dic[note_to_int.get(("0.75", False))] = 0.05   #dotted eighth note
        prob_dic[note_to_int.get(("1.5", True))] = 0.05     #dotted quarter rest
        prob_dic[note_to_int.get(("1.5", False))] = 0.1     #dotted quarter note
        prob_dic[note_to_int.get(("0.25", True))] = 0.05    #sixteenth rest
        prob_dic[note_to_int.get(("0.25", False))] = 0.1    #sixteenth note
        prob_dic[note_to_int.get(("0.5", True))] = 0.05     #eighth rest
        prob_dic[note_to_int.get(("0.5", False))] = 0.1     #eighth note
        prob_dic[note_to_int.get(("1.0", True))] = 0.05     #quarter rest
        prob_dic[note_to_int.get(("1.0", False))] = 0.1     #quarter note
        prob_dic[note_to_int.get(("2.0", True))] = 0.025    #half rest
        prob_dic[note_to_int.get(("2.0", False))] = 0.05    #half note
        return prob_dic
    elif level == 9:
        prob_dic = dict()
        prob_dic[note_to_int.get(("1/3", True))] = 0.1      #triplet eighth rest
        prob_dic[note_to_int.get(("1/3", False))] = 0.15    #triplet eighth note
        prob_dic[note_to_int.get(("0.75", True))] = 0.05    #dotted eighth rest
        prob_dic[note_to_int.get(("0.75", False))] = 0.05   #dotted eighth note
        prob_dic[note_to_int.get(("1.5", True))] = 0.05     #dotted quarter rest
        prob_dic[note_to_int.get(("1.5", False))] = 0.05    #dotted quarter note
        prob_dic[note_to_int.get(("0.25", True))] = 0.1     #sixteenth rest
        prob_dic[note_to_int.get(("0.25", False))] = 0.15   #sixteenth note
        prob_dic[note_to_int.get(("0.5", True))] = 0.05     #eighth rest
        prob_dic[note_to_int.get(("0.5", False))] = 0.1     #eighth note
        prob_dic[note_to_int.get(("1.0", True))] = 0.05     #quarter rest
        prob_dic[note_to_int.get(("1.0", False))] = 0.05    #quarter note
        prob_dic[note_to_int.get(("2.0", True))] = 0.025    #half rest
        prob_dic[note_to_int.get(("2.0", False))] = 0.025   #half note
        return prob_dic
    elif level == 10:
        prob_dic = dict()
        prob_dic[note_to_int.get(("1/12", True))] = 0.1      #triplet sixteenth rest
        prob_dic[note_to_int.get(("1/12", False))] = 0.1    #triplet sixteenth note
        prob_dic[note_to_int.get(("2/3", True))] = 0.05      #triplet quarter rest
        prob_dic[note_to_int.get(("2/3", False))] = 0.1    #triplet quarter note


        prob_dic[note_to_int.get(("1/3", True))] = 0.05      #triplet eighth rest
        prob_dic[note_to_int.get(("1/3", False))] = 0.1    #triplet eighth note
        prob_dic[note_to_int.get(("0.75", True))] = 0.025    #dotted eighth rest
        prob_dic[note_to_int.get(("0.75", False))] = 0.025   #dotted eighth note
        prob_dic[note_to_int.get(("1.5", True))] = 0.025     #dotted quarter rest
        prob_dic[note_to_int.get(("1.5", False))] = 0.05    #dotted quarter note
        prob_dic[note_to_int.get(("0.25", True))] = 0.05     #sixteenth rest
        prob_dic[note_to_int.get(("0.25", False))] = 0.1   #sixteenth note
        prob_dic[note_to_int.get(("0.5", True))] = 0.05     #eighth rest
        prob_dic[note_to_int.get(("0.5", False))] = 0.1     #eighth note
        prob_dic[note_to_int.get(("1.0", True))] = 0.025     #quarter rest
        prob_dic[note_to_int.get(("1.0", False))] = 0.025    #quarter note
        prob_dic[note_to_int.get(("2.0", True))] = 0.0125    #half rest
        prob_dic[note_to_int.get(("2.0", False))] = 0.0125   #half note
        return prob_dic
    else:
        raise Exception("Level " + level + " is not a valid level")

def create_rand_tempo():
    """Create a random tempo between 40 and 130"""
    return np.random.randint(40, high=130)

def create_rand_notes(level, tempo_time):
    """Create an array of 50 random notes"""
    rand_note_dist = get_prop_level(level)

    # Get all possible notes for the level, and the PDF of probabilites for them to appear.
    sum = 0.0
    ids = []
    distribution = []
    for id, prob in rand_note_dist.items():
        ids.append(id)
        distribution.append(prob)
        sum += prob

    #print(sum)

    # Using the tempo, we will get enough of these notes to come within +-10 seconds of a
    # minutes worth of notes.
    offset = 0.0
    upper_limit = tempo_time + 10.0
    lower_limit = tempo_time - 10.0
    num_notes = 50
    while offset <= lower_limit or offset >= upper_limit:
        offset = 0.0
        rand_notes = np.random.choice(ids, p=distribution, size=num_notes)
        for n in rand_notes:
            offset += float(int_to_note.get(n)[0])

        if offset <= lower_limit:
            num_notes += 5
        elif offset >= upper_limit:
            num_notes -= 5

    return rand_notes

def create_midi(notes, tempo_time, file_name):
    """Given an array of notes, a tempo to use, and a file name, create a midi music track"""
    output_notes = []
    output_notes.append(tempo.MetronomeMark(number=tempo_time))
    int_to_note = get_note_dic(True)
    offset = 0.0
    for id in notes:
        if int_to_note.get(id) != "|":
            length, is_rest = int_to_note.get(id)
            new_note = create_note(float(length), is_rest)
            new_note.offset = offset
            output_notes.append(new_note)

            # increase offset each iteration so that notes do not stack
            offset += new_note.duration.quarterLength

    #print(offset)
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='{}.mid'.format(file_name))

def create_text_file(notes, tempo_time, file_name):
    """Create the text file for machine learning input"""
    str_notes = [str(x) for x in notes]
    str_notes.append("0")
    with open("{}.txt".format(file_name), "w") as file:
        file.write(str(tempo_time) + ":")
        file.write("\t".join(str_notes))

def print_notes():
    """Print out the complete list of notes and their ids"""
    for id, n in get_note_dic(True).items():
        print("Note id: " + str(id))
        if id == 0:
            print(n)
        else:
            new_note = create_note(n[0], n[1])
            print("Is rest: " + str(new_note.isRest))
            print("Note type: " + new_note.duration.type)
            print("Note length: " + str(new_note.duration.quarterLength))

        print()

def create_example(level):
    """
    Create an example with a random tempo and random set of notes based on
    a given level. This makes both the midi file and the corresponding text output
    file that the program would use.
    """
    rand_tempo = create_rand_tempo()
    rand_notes = create_rand_notes(level, rand_tempo)
    create_midi(rand_notes, rand_tempo, "random_{}".format(level))
    create_text_file(rand_notes, rand_tempo, "random_{}".format(level))

def create_output_files(level, num_files):
    """
    Create a set of text files that the program will use to play the game and
    the ML algorithm will use to train on.
    """
    for file in range(num_files):
        rand_tempo = create_rand_tempo()
        rand_notes = create_rand_notes(level, rand_tempo)
        create_text_file(rand_notes, rand_tempo, "{}_{}".format(level, file))

def convert_text_to_midi(dir):
    """
    Takes a directory of txt files that are correctly formatted, and creates midi
    files based off of them.
    """
    for glob_file in glob.glob("{}/*.txt".format(dir)):
        print(glob_file)
        with open(glob_file, "r") as file:
            text = file.read()

        if re.match(r"^\d+:(\d+\t)+0$", text):
            file_name = os.path.split(glob_file)[1].split(".")[0]
            parts = text.split(":")
            tempo_time = int(parts[0])
            notes = [int(x) for x in parts[1].split("\t")]
            create_midi(notes, tempo_time, file_name)



def convert_midi_to_text(dir):
    """
    Takes a directory of midi files, and creates text files based off of them.
    """
    note_to_int = get_note_dic(False)
    for glob_file in glob.glob("{}/*.mid".format(dir)):
        midi = converter.parse(glob_file)
        notes = []
        tempo_time = 0
        file_name = os.path.split(glob_file)[1].split(".")[0]

        print(glob_file)

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            score = s2.parts[0].recurse() #.makeRests(fillGaps=True, inPlace=False)
            #tempo_time = s2.flat.getElementsByClass(tempo.MetronomeMark)[0].number
        except: # file has notes in a flat structure
            score = midi.flat.recurse()
            #tempo_time = midi.flat.getElementsByClass(tempo.MetronomeMark)[0].number

        def set_note(element, is_rest):
            note_parts = element.splitAtDurations()
            for np in note_parts:
                temp_length = np.duration.quarterLength
                num_whole_notes = 0
                while temp_length > 4.0:
                    temp_length -= 4.0
                    num_whole_notes += 1

                if num_whole_notes > 0:
                    notes.append(note_to_int.get((str(temp_length), is_rest)))
                    for _ in range(num_whole_notes):
                        notes.append(note_to_int.get(("4.0", is_rest)))
                else:
                    #print(np.duration.type)
                    #print(np.offset)
                    #print((temp_length, is_rest))
                    #print(note_to_int.get((str(temp_length), is_rest)))
                    notes.append(note_to_int.get((str(temp_length), is_rest)))

        for element in score:
            if isinstance(element, note.Note):
                set_note(element, False)
            elif isinstance(element, note.Rest):
                set_note(element, True)
            elif isinstance(element, tempo.MetronomeMark):
                tempo_time = int(element.number)

        create_text_file(notes, tempo_time, file_name)


def get_data_from_midi(dir):
    """
    Takes a directory holding midi files and outputs matching length lists of
    level numbers and music sequences.
    """
    note_to_int = get_note_dic(False)
    sequences = []
    levels = []
    for glob_file in glob.glob("{}/*.mid".format(dir)):
        midi = converter.parse(glob_file)
        notes = []
        tempo_time = 0
        file_name = os.path.split(glob_file)[1].split(".")[0]

        levels.append(int(file_name.split("_")[0]) - 1)

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            score = s2.recurse() #.makeRests(fillGaps=True, inPlace=False)
            #tempo_time = s2.flat.getElementsByClass(tempo.MetronomeMark)[0].number
        except: # file has notes in a flat structure
            score = midi.recurse()
            #tempo_time = midi.flat.getElementsByClass(tempo.MetronomeMark)[0].number

        def set_note(element, is_rest):
            note_parts = element.splitAtDurations()
            for np in note_parts:
                temp_length = np.duration.quarterLength
                num_whole_notes = 0
                while temp_length > 4.0:
                    temp_length -= 4.0
                    num_whole_notes += 1

                if num_whole_notes > 0:
                    if note_to_int.get((str(temp_length), is_rest)) == None:
                        continue

                    notes.append(note_to_int.get((str(temp_length), is_rest)))
                    for _ in range(num_whole_notes):
                        notes.append(note_to_int.get(("4.0", is_rest)))
                else:
                    if note_to_int.get((str(temp_length), is_rest)) == None:
                        continue
                    notes.append(note_to_int.get((str(temp_length), is_rest)))

        for element in score:
            if isinstance(element, note.Note):
                set_note(element, False)
            elif isinstance(element, note.Rest):
                set_note(element, True)
            elif isinstance(element, tempo.MetronomeMark):
                tempo_time = int(element.number)

        sequences.append(notes)

    return levels, sequences


if __name__ == '__main__':
    #print_notes()
    #create_example(10)
    #create_output_files(10, 15)
    #convert_text_to_midi(".")
    convert_midi_to_text(".")
