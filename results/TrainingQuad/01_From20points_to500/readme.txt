Predtim se mi dron choval tak, ze byl zaseklej na 20 bodech, protoze jsem mel moc velkou mapu a on nevedel, co znamena input fire_pos_x/y, takze neletal za ohnem. Ted jsme zmensili velikost mapy na 500x500 metru, diky cemuz dron nasel ohen mnohem rychleji a tim si spojil tuto informaci.
Problem je s kritikem, ktery byl predtim natrenovan na 20 bodu max a ted se mu tam objevuji hodnoty ve stovkach az tisicu.
Slo by to budto cele pretrenovat, nebo ho nechat trenovat mnohem dele, aby se kritik pretrenoval.
Kdyz se podivam na odmeny, tak jde videt, ze mame clipping na reward 10, takze by odmena byla mnohem vetsi.
Zkusim ted udelat to, ze drona pretrenujeme uplne,le snizime o dva rady reward za videni ohne, at to stahneme dolu.
Diky tomu by se mel kritik naucit rovnou, s jakymi odmenami fungujeme a rovnez by dron mohl vyhledavat ohen mnohem lepe.
Zkusim pridat jeste penalizaci za moc nizky let, ktery je nebezpecny.
Kdyz se podivam na 1000x1000 tak jde jasne videt jak odmena za ohen totalne prebije vysku, takze dron leti nahoru a vubec ho nezajima, ze je sto metru od ohne. To je spatne. Nechceme mit jednoho drone, kterej vyleti kilometr nahoru a bude sledovat celou mapu.


Commit: 7fe4ca29137079419dd422f4c4df653303a02b1f
Commit message: Quad finally working again.
Training with adaptive camera and demo working too.