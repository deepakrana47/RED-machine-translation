# import pickle, re
# from utility import extract_feature_using_senna
# # wrd = extract_feature_using_senna('She had been critically ill since having surgery at Baptist Hospital on May 7 to replace a heart valve')
# wrd = extract_feature_using_senna('U.N. inspectors found traces of highly enriched weapons-grade uranium at an Iranian nuclear facility, a report by the U.N. nuclear agency says')
# # wrd = extract_feature_using_senna('United Nations inspectors have discovered traces of highly enriched uranium near an Iranian nuclear facility heightening worries that the country may have a secret nuclear weapons program')
# # wrd = extract_feature_using_senna('United Nations inspectors have discovered traces of highly enriched uranium near an Iranian nuclear facility, heightening worries that the country may have a secret nuclear weapons program')
# # print wrd
#
#
# # def syn_processing(wrd):
# #     wd = {}
# #     for i in wrd:
# #         if type(i) == int:
# #             wd[i] = wrd[i]
# #
# #     t = re.sub(r'([\)\(])', r' \1 ', wrd['tree'])
# #     t = re.sub(r'[ ]{2,}', r' ', t)
# #     t = t.split(' ')
# #     p = []
# #     id = 1
# #     ind = 0
# #     h_vect = []
# #     h_index = {}
# #
# #     for i in range(len(t)):
# #         if t[i] != '(' and t[i] != ')' and t[i] != '' and t[i] == wrd[id]['word']:
# #             temp = id
# #             h_index[temp] = ind
# #             id += 1
# #             ind += 1
# #             t[i] = temp
# #             p.append(wrd[temp]['pid'])
# #         else:
# #             p.append(t[i])
# #
# #     while i < len(p):
# #         if p[i] == -1:
# #             p.pop(i)
# #             t.pop(i)
# #         i += 1
# #
# #     # check for redudent brackets
# #     bs = [];
# #     be = []
# #     i = 0
# #     while i < len(t):
# #         if t[i] == '(':
# #             bs.append(i)
# #         elif t[i] == ')':
# #             be.append(i)
# #         if len(be) == 2:
# #             if bs[-1] - bs[-2] == 1 and be[1] - be[0] == 1:
# #                 t.pop(be[0])
# #                 t.pop(bs[-1])
# #                 be[1] -= 2
# #                 bs[-1] -= 1
# #                 i -= 2
# #             a = be.pop(0)
# #             for j in sorted(range(len(bs)), reverse=1):
# #                 if bs[j] < a:
# #                     bs.pop(j)
# #                     break
# #         i += 1
# #     stack = []
# #     i = 0
# #     h_ind = 0
# #     while len(t) > 1:
# #         if t[i] is '':
# #             t.pop(i)
# #             continue
# #         elif t[i] is '(':
# #             stack.append(i)
# #         elif t[i] is ')':
# #             h_vect.append([])
# #             for j in range(stack[-1] + 1, i):
# #                 h_vect[h_ind].append(h_index[t[j]])
# #             h_index[id] = ind
# #             for l in range(stack[-1], i):
# #                 t.pop(stack[-1])
# #             t[stack[-1]] = id
# #             i = stack[-1]
# #             stack.pop()
# #             if not h_vect[h_ind]:
# #                 i += 1
# #                 continue
# #             h_ind += 1
# #             ind += 1
# #             id += 1
# #         i += 1
# #
# #     wp = []
# #     for i in range(len(h_vect)):
# #         wp.append([])
# #         count = len(h_vect[i]) - 1
# #         for j in h_vect[i]:
# #             wp[i].append(count)
# #             count -= 1
# #     return h_index, h_vect, wp
# #
# # _, h_vect, wp = syn_processing(wrd)
# # print h_vect
# # print wp
#
#
# def removing_stopword( wd, stopwds):
#     wd_index = [i for i in wd]
#     p = []
#     for i in sorted(wd):
#         p.append(wd[i]['pid'])
#     p = sorted(list(set(p)))
#     for i in wd_index:
#         if i not in p and wd[i]['word'] in stopwds:
#             wd.pop(i)
#     return wd
#
# stopwds = [i.strip('\n') for i in open('/media/zero/41FF48D81730BD9B/DT_RAE/Chunk_rnn/config/stopword.txt', 'r')]
# wd = {}; oth={}
# for i in wrd:
#     if type(i) == int:
#         wd[i] = wrd[i]
#     else:
#         oth[i] = wrd[i]
# wrd = removing_stopword(wd, stopwds)
# wrd.update(oth)
#
# def syn_processing2(wrd):
#     wd = {}
#     for i in wrd:
#         if type(i) == int:
#             wd[i] = wrd[i]
#
#     t = re.sub(r'([\)\(])', r' \1 ', wrd['tree'])
#     t = re.sub(r'[ ]{2,}', r' ', t)
#     t = t.split(' ')
#     p = []
#     wids = sorted([i for i in wd])
#     ind = 0
#     h_vect = []
#     h_index = {}
#
#     id=wids.pop(0)
#     for i in range(len(t)):
#         if t[i] != '(' and t[i] != ')' and t[i] != '' and t[i] == wrd[id]['word']:
#             temp = id
#             h_index[temp] = ind
#             id += 1
#             ind += 1
#             t[i] = temp
#             p.append(wrd[temp]['pid'])
#         else:
#             p.append(t[i])
#
#     while i < len(p):
#         if p[i] == -1:
#             p.pop(i)
#             t.pop(i)
#         i += 1
#
#     # check for redudent brackets
#     bs = [];
#     be = []
#     i = 0
#     while i < len(t):
#         if t[i] == '(':
#             bs.append(i)
#         elif t[i] == ')':
#             be.append(i)
#         if len(be) == 2:
#             if bs[-1] - bs[-2] == 1 and be[1] - be[0] == 1:
#                 t.pop(be[0])
#                 t.pop(bs[-1])
#                 be[1] -= 2
#                 bs[-1] -= 1
#                 i -= 2
#             a = be.pop(0)
#             for j in sorted(range(len(bs)), reverse=1):
#                 if bs[j] < a:
#                     bs.pop(j)
#                     break
#         i += 1
#     stack = []
#     i = 0
#     h_ind = 0
#     while len(t) > 1:
#         if t[i] is '':
#             t.pop(i)
#             continue
#         elif t[i] is '(':
#             stack.append(i)
#         elif t[i] is ')':
#             h_vect.append([])
#             for j in range(stack[-1] + 1, i):
#                 h_vect[h_ind].append(h_index[t[j]])
#             h_index[id] = ind
#             for l in range(stack[-1], i):
#                 t.pop(stack[-1])
#             t[stack[-1]] = id
#             i = stack[-1]
#             stack.pop()
#             if not h_vect[h_ind]:
#                 i += 1
#                 continue
#             h_ind += 1
#             ind += 1
#             id += 1
#         i += 1
#
#     wp = []
#     for i in range(len(h_vect)):
#         wp.append([])
#         count = len(h_vect[i]) - 1
#         for j in h_vect[i]:
#             wp[i].append(count)
#             count -= 1
#     return h_index, h_vect, wp
#
# _, h_vect, wp = syn_processing2(wrd)
# print h_vect
# print wp

import numpy as np
x= np.array([[ 634.70938763], [ 175.94219013], [ 272.30516291], [ 303.58359369], [ 460.12055249], [ 212.91627179], [ 527.76580344], [ 317.69205774], [ 129.63317218], [ 714.09535767], [ 575.56424919], [ 247.85648422], [ 315.9419359], [ 420.00012245], [ 115.7176693], [ 80.23112276], [ 215.33319628], [ 289.31724283], [ 221.37973088], [ 466.33107851], [ 262.20324468], [ 289.51668153], [ 280.53824677], [ 217.23584361], [ 411.36861037], [ 651.80957874], [-1.], [ 272.6074687], [ 148.49511199], [ 79.67357749], [ 252.93801539], [ 339.1598914], [ 431.39817499], [ 510.76363884], [ 422.64527847], [ 148.68419097], [ 135.69261145], [ 43.0059753], [ 367.29204234], [ 17.68645027], [-0.80375095], [ 489.16690642], [ 465.89907142], [ 280.72393459], [ 34.57255107], [ 195.31397911], [ 305.62582979], [ 319.77242136], [ 310.73224347], [ 320.65815144]])
# np.where(x > 0, x, np.exp(x) - np.ones(x.shape))
# for i in range(len(x)):
#     if x[i] > 0:
#         print i,x[i]
#     else:
#         print i,np.exp(x[i])-1

print np.array([x[i] if x[i] > 0 else np.exp(x[i])-1 for i in range(len(x)) ])