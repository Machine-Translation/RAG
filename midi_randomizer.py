# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 20:00:10 2019

@author: Cory
"""


from music21 import converter, instrument, chord, stream, tempo, note
import numpy as np

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
    EX: 2.5 = dotted half note
    EX: 1/3 = an eigth note in a triplet
    """
    lengths = [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.75, 2/3, 0.5, 1/3, 0.25, 1/12]
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
        prob_dic[note_to_int.get((1.0, True))] = 0.2  #quarter rest
        prob_dic[note_to_int.get((1.0, False))] = 0.5 #quarter note
        prob_dic[note_to_int.get((2.0, True))] = 0.1  #half rest
        prob_dic[note_to_int.get((2.0, False))] = 0.2 #half note
        return prob_dic
    elif level == 2:
        prob_dic = dict()
        prob_dic[note_to_int.get((0.5, True))] = 0.1  #eigth rest
        prob_dic[note_to_int.get((0.5, False))] = 0.4 #eigth note
        prob_dic[note_to_int.get((1.0, True))] = 0.1  #quarter rest
        prob_dic[note_to_int.get((1.0, False))] = 0.2 #quarter note
        prob_dic[note_to_int.get((2.0, True))] = 0.1  #half rest
        prob_dic[note_to_int.get((2.0, False))] = 0.1 #half note
        return prob_dic
    elif level == 3:
        prob_dic = dict()
        prob_dic[note_to_int.get((0.25, True))] = 0.05     #sixteenth rest
        prob_dic[note_to_int.get((0.25, False))] = 0.05    #sixteenth note
        prob_dic[note_to_int.get((0.5, True))] = 0.15     #eigth rest
        prob_dic[note_to_int.get((0.5, False))] = 0.3     #eigth note
        prob_dic[note_to_int.get((1.0, True))] = 0.15     #quarter rest
        prob_dic[note_to_int.get((1.0, False))] = 0.2     #quarter note
        prob_dic[note_to_int.get((2.0, True))] = 0.05     #half rest
        prob_dic[note_to_int.get((2.0, False))] = 0.05    #half note
        return prob_dic
    elif level == 4:
        prob_dic = dict()
        prob_dic[note_to_int.get((0.25, True))] = 0.2      #sixteenth rest
        prob_dic[note_to_int.get((0.25, False))] = 0.4     #sixteenth note
        prob_dic[note_to_int.get((0.5, True))] = 0.1      #eigth rest
        prob_dic[note_to_int.get((0.5, False))] = 0.15    #eigth note
        prob_dic[note_to_int.get((1.0, True))] = 0.05     #quarter rest
        prob_dic[note_to_int.get((1.0, False))] = 0.05    #quarter note
        prob_dic[note_to_int.get((2.0, True))] = 0.025    #half rest
        prob_dic[note_to_int.get((2.0, False))] = 0.025   #half note
        return prob_dic
    elif level == 5:
        prob_dic = dict()
        prob_dic[note_to_int.get((0.25, True))] = 0.2     #sixteenth rest
        prob_dic[note_to_int.get((0.25, False))] = 0.3    #sixteenth note
        prob_dic[note_to_int.get((0.5, True))] = 0.1      #eigth rest
        prob_dic[note_to_int.get((0.5, False))] = 0.2     #eigth note
        prob_dic[note_to_int.get((1.0, True))] = 0.05     #quarter rest
        prob_dic[note_to_int.get((1.0, False))] = 0.1     #quarter note
        prob_dic[note_to_int.get((2.0, True))] = 0.025    #half rest
        prob_dic[note_to_int.get((2.0, False))] = 0.025   #half note
        return prob_dic
    elif level == 6:
        prob_dic = dict()
        prob_dic[note_to_int.get((1.5, True))] = 0.05     #dotted quarter rest
        prob_dic[note_to_int.get((1.5, False))] = 0.25     #dotted quarter note
        prob_dic[note_to_int.get((0.25, True))] = 0.05     #sixteenth rest
        prob_dic[note_to_int.get((0.25, False))] = 0.2    #sixteenth note
        prob_dic[note_to_int.get((0.5, True))] = 0.05      #eigth rest
        prob_dic[note_to_int.get((0.5, False))] = 0.2    #eigth note
        prob_dic[note_to_int.get((1.0, True))] = 0.05     #quarter rest
        prob_dic[note_to_int.get((1.0, False))] = 0.1     #quarter note
        prob_dic[note_to_int.get((2.0, True))] = 0.025    #half rest
        prob_dic[note_to_int.get((2.0, False))] = 0.025   #half note
        return prob_dic
    else:
        raise Exception("Level " + level + " is not a valid level")

def create_rand_tempo():
    """Create a random tempo between 40 and 130"""
    return np.random.randint(40, high=130)

def create_rand_notes(level):
    """Create an array of 50 random notes"""
    rand_note_dist = get_prop_level(level)

    ids = []
    distribution = []
    for id, prob in rand_note_dist.items():
        ids.append(id)
        distribution.append(prob)

    return np.random.choice(ids, p=distribution, size=50)

def create_rand_midi(notes, tempo_time, file_name):
    """Create a random music track from midi given a music level"""
    output_notes = []
    output_notes.append(tempo.MetronomeMark(number=tempo_time))
    int_to_note = get_note_dic(True)
    offset = 0.0
    for id in notes:
        length, is_rest = int_to_note.get(id)
        new_note = create_note(length, is_rest)
        new_note.offset = offset
        output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += new_note.duration.quarterLength

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='{}.mid'.format(file_name))

def create_text_file(notes, tempo_time, file_name):
    """Create the text file for machine learning input"""
    str_notes = [str(x) for x in notes]
    str_notes.append("0")
    with open("{}.txt".format(file_name), "w") as file:
        file.write(str(tempo_time) + ":")
        file.write("\t".join(str_notes))


if __name__ == '__main__':
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

    level = 6
    rand_tempo = create_rand_tempo()
    rand_notes = create_rand_notes(level)
    create_rand_midi(rand_notes, rand_tempo, "random_{}".format(level))
    create_text_file(rand_notes, rand_tempo, "random_{}".format(level))