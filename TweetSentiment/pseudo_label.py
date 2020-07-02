models = []

INDEX = []
DISPLAY =  1
for i in range(len(models)):
    preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))
    preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))
    K.clear_session()

    model, padded_model = build_model()
    load_weights(model, models[i])
    preds = padded_model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)
    preds_start = preds[0]
    preds_end = preds[1]
    for k in range(input_ids_t.shape[0]):
        a = np.argmax(preds_start[k,])
        b = np.argmax(preds_end[k,])
        INDEX.append((a,b))

from collections import Counter

ALL = []
for i in range(0,len(INDEX),3534):
    ALL.append(INDEX[i:i+3534])

ind1 = ALL[0]
ind2 = ALL[1]
ind3 = ALL[2]
ind4 = ALL[3]
ind5 = ALL[4]

final_ind = []
for i in range(len(ind1)):
    if ind1[i] == ind2[i] and ind2[i] == ind3[i] and ind3[i] == ind4[i] and ind4[i] == ind5[i]:
        final_ind.append(i)

fake_ids = np.ones((len(final_ind),MAX_LEN),dtype='int32')
fake_attention_mask = np.zeros((len(final_ind),MAX_LEN),dtype='int32')
fake_token_type_ids = np.zeros((len(final_ind),MAX_LEN),dtype='int32')

for i in range(len(final_ind)):
    j = final_ind[i]
    fake_ids[i] = input_ids_t[j]
    fake_attention_mask[i] = attention_mask_t[j]
    fake_token_type_ids[i] = token_type_ids_t[j]


fake_start_tokens = np.zeros((len(final_ind),MAX_LEN),dtype='int32')
fake_end_tokens = np.zeros((len(final_ind),MAX_LEN),dtype='int32')

for i in range(len(final_ind)):
    j = final_ind[i]
    s = ind1[j][0]
    e = ind1[j][1]
    fake_start_tokens[i][s] = 1
    fake_end_tokens[i][e] = 1
