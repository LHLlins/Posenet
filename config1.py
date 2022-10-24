from os import kill


id = ""
henrique_counter = 0
id_len = [0, 0, 0, 0, 0, 0, 0, 0]
altura_id = 0
kill_cam = False
arduino = 0



def person_id(vector_counter, id):
    for i in range(len(vector_counter)):
        if i > 30:
            return id
    return "Id not defined"

def counter_id(id_len,id):

    if id=="Henrique":
        id_len[0] +=1
    elif id=="Lazaro":
        id_len[1] += 1
    elif id=="Guilherme":
        id_len[2] += 1
    elif id=="Arthur":
        id_len[3] += 1
    elif id=="Bruno B":
        id_len[4] += 1
    elif id=="Flavio":
        id_len[5] += 1
    elif id=="Augusto":
        id_len[6] += 1    
    elif id=="Carlos Augusto":
        id_len[7] += 1 
    return id_len

def altura(id, altura):

    if id == "Henrique":
         altura = 184
         return altura
    elif id == "Lazaro":
        altura = 178
        return altura
    elif id == "Guilherme":
        altura = 173
        return altura 
    elif id == "Arthur":
        altura = 184
        return altura
    elif id == "Bruno B":
        altura = 185
        return altura 
    elif id == "Flavio":
        altura = 175
    elif id == "Augusto":
        altura = 172
        return altura
    elif id=="Carlos Augusto":
        altura = 168
        return altura       
    else:
        return "Not defined"







