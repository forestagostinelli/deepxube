edge_or_corner(Cbl) :- edge_cbl(Cbl).
edge_or_corner(Cbl) :- corner_cbl(Cbl).
cross(F, CrossCol) :- face(F), color(CrossCol), #count{Cbl: edge_cbl(Cbl), onface(Cbl, CrossCol, F)} = 4.
x(F, XCol) :- face(F), color(XCol), #count{Cbl: corner_cbl(Cbl), onface(Cbl, XCol, F)} = 4.
checkerboard(F) :- face(F), x(F, ColX), face_col(F, ColX), cross(F, ColCr), dif_col(ColX, ColCr).
cup(F1, F2, CCol) :- dif_face(F1, F2), face_col(F1, F1Col), dif_col(F1Col, CCol), edge_cbl(ECbl), onface(ECbl, _, F2), onface(ECbl, F1Col, F1), #count{Cbl: edge_or_corner(Cbl), onface(Cbl, CCol, F1)} = 7.
spot(F, BCol) :- color(BCol), face(F), face_col(F, FCol), dif_col(FCol, BCol), #count{Cbl: onface(Cbl, BCol, F), edge_or_corner(Cbl)} = 8.
face_same(F) :- face_col(F, FCol), #count{Cbl : onface(Cbl, FCol, F)}=9.
not_group180 :- onface(Cbl, StkCol, Fon), face_col(F, StkCol), face_adj(F, Fon).
group180 :- onface(wg_c, w, w_f), not not_group180.


% cube2(F0, F1, F2) :- face_rel(F0, F1, F2, _), edge_cbl(ECbl), onface(ECbl, _, F0), onface(ECbl, _, F1), in_place(ECbl).
% cube2(F0, F1, F2) :- dif_face(F0, F1), dif_face(F0, F2), dif_face(F1, F2), face_col(F0, F0Col), face_col(F1, F1Col), face_col(F2, F2Col), #count{edge_cbl(Cbl0), has_stk_col(Cbl0, F0Col), in_place(Cbl)} =

failure0 :- onface(Cbl1, r, w_f), center_cbl(Cbl1), onface(Cbl2, w, g_f), center_cbl(Cbl2).
failure1 :- onface(Cbl, Col1, F1), onface(Cbl, Col1, F2), center_cbl(Cbl), white(Col1), face_col(F1, FCol1), face_col(F2, FCol2), white(FCol1), red(FCol2).
failure2 :- onface(Cbl,Col1,F2),center_cbl(Cbl).
failure3 :- cubelet(V0),has_stk_col(V0,V1),onface(V0,V1,V2).
long1 :- cubelet(V0),has_stk_col(V0,V1),onface(V0,V1,V2),has_stk_col(V0,V3),face_col(V2,V4),cubelet(V5),has_stk_col(V5,V6),onface(V5,V1,V7).
% long2 :- cubelet(V0),has_stk_col(V0,V1),onface(V0,V1,V2),has_stk_col(V0,V3),face_col(V2,V4),cubelet(V5),has_stk_col(V5,V6),onface(V5,V1,V7),cubelet(V8),has_stk_col(V8,V4),onface(V8,V6,V9).
% long3 :- cubelet(V0),has_stk_col(V0,V1),onface(V0,V1,V2),onface(V0,V1,V3),cubelet(V4),has_stk_col(V4,V1),onface(V4,V1,V2),cubelet(V5),has_stk_col(V5,V6),onface(V5,V6,V7),cubelet(V8),has_stk_col(V8,V6),onface(V8,V6,V9).

cupup :- cup(r_f, w_f, _), cup(o_f, w_f, _), cup(b_f, w_f, _), cup(g_f, w_f, _).
x6 :- #count{F: x(F, XCol), face_col(F, XCol)} = 6.
cross6 :- #count{F: cross(F, CCol), face_col(F, CCol)} = 6.
xwhite :- x(w_f, w).
crosswhite :- cross(w_f, w), #count{Cbl: in_place(Cbl), center_cbl(Cbl)}=6.
cupplus :- cup(F1, w_f, _), cup(F2, w_f, _), face_rel(_, F1, F2, cl), cup(F3, y_f, _), cup(F4, y_f, _), face_rel(_, F3, F4, cl).
cupring :- cup(r_f, _, _), cup(o_f, _, _), cup(b_f, _, _), cup(g_f, _, _).
checker6 :- #count{F: checkerboard(F)} = 6.
checkerop :- checkerboard(F1), checkerboard(F2), face_rel(_, F1, F2, op), #count{F: face_same(F)} = 4.

spotcup :- spot(F1, _), cup(F2, F1, _), face_rel(_, F1, F2, cl).
canon :- #count{F : face_same(F)}=6.