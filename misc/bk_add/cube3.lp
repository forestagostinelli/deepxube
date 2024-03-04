edge_or_corner(Cbl) :- edge_cbl(Cbl).
edge_or_corner(Cbl) :- corner_cbl(Cbl).
cross(F, CrossCol) :- face(F), color(CrossCol), #count{Cbl: edge_cbl(Cbl), onface(Cbl, CrossCol, F)} = 4.
x(F, XCol) :- face(F), color(XCol), #count{Cbl: corner_cbl(Cbl), onface(Cbl, XCol, F)} = 4.
checkerboard(F) :- face(F), x(F, ColX), face_col(F, ColX), cross(F, ColCr), dif_col(ColX, ColCr).
cup(F1, F2, CCol) :- dif_face(F1, F2), face_col(F1, F1Col), dif_col(F1Col, CCol), edge_cbl(ECbl), onface(ECbl, _, F2), onface(ECbl, F1Col, F1), #count{Cbl: edge_or_corner(Cbl), onface(Cbl, CCol, F1)} = 7.
spot(F, BCol) :- color(BCol), face(F), face_col(F, FCol), dif_col(FCol, BCol), #count{Cbl: onface(Cbl, BCol, F), edge_or_corner(Cbl)} = 8.
face_same(F) :- face_col(F, FCol), #count{Cbl : onface(Cbl, FCol, F)}=9.
has_at_least_1_same(F) :- onface(_, Col, F), face_col(F, Col).

cupup :- cup(r_f, w_f, _), cup(o_f, w_f, _), cup(b_f, w_f, _), cup(g_f, w_f, _).
x6 :- #count{F: x(F, XCol), face_col(F, XCol)} = 6.
cross6 :- #count{F: cross(F, CCol), face_col(F, CCol)} = 6.
cupplus :- cup(F1, w_f, _), cup(F2, w_f, _), face_rel(_, F1, F2, cl), cup(F3, y_f, _), cup(F4, y_f, _), face_rel(_, F3, F4, cl).
cupring :- cup(r_f, _, _), cup(o_f, _, _), cup(b_f, _, _), cup(g_f, _, _).
checker6 :- #count{F: checkerboard(F)} = 6.
checkerop :- checkerboard(F1), checkerboard(F2), face_rel(_, F1, F2, op), #count{F: face_same(F)} = 4.

spotcup :- spot(F1, _), cup(F2, F1, _), face_rel(_, F1, F2, cl).
canon :- #count{F : face_same(F)}=6.