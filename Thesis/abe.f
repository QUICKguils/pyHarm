   subroutine ApprxBifEqua(n,Yb,H1,H2,Hs,a,b,c)



        integer        , intent(IN) :: n

        real(kind=long), intent(IN) :: Yb(:),H1(:),H2(:),Hs(:)

        real(kind=long), intent(OUT):: a,b,c



        integer            :: iErr

        real(kind=long),allocatable,save :: Y1(:),F(:)

        real(kind=long)    :: normF, g0, g1, g2, g3, g4, g5, g6, g7, g8, eps=1.0D-6



        if (.not.allocated(Y1)) then

            allocate(Y1(n+1), F(n), stat = iErr)

            call ErrorStop('ApprxBifEqua: Allocation error with Y1 and F: iErr=',iErr)

        endif



        !calculate a,b,c

        !0 - g(0,0)

        call FEqsBfc(n,Yb,F,normF,iErr)

        g0 = dot_product(Hs,F)

        !1 - g(eps,0)

        Y1 = Yb+eps*H1

        call FEqsBfc(n,Y1,F,normF,iErr)

        g1 = dot_product(Hs,F)

        !2 - g(0,eps)

        Y1 = Yb+eps*H2

        call FEqsBfc(n,Y1,F,normF,iErr)

        g2 = dot_product(Hs,F)

        !3 - g(-eps,0)

        Y1 = Yb-eps*H1

        call FEqsBfc(n,Y1,F,normF,iErr)

        g3 = dot_product(Hs,F)

        !4 - g(0,-eps)

        Y1 = Yb-eps*H2

        call FEqsBfc(n,Y1,F,normF,iErr)

        g4 = dot_product(Hs,F)

        !5 - g(eps,eps)

        Y1 = Yb+eps*H1+eps*H2

        call FEqsBfc(n,Y1,F,normF,iErr)

        g5 = dot_product(Hs,F)

        !6 - g(-eps,-eps)

        Y1 = Yb-eps*H1-eps*H2

        call FEqsBfc(n,Y1,F,normF,iErr)

        g6 = dot_product(Hs,F)

        !7 - g(eps,-eps)

        Y1 = Yb+eps*H1-eps*H2

        call FEqsBfc(n,Y1,F,normF,iErr)

        g7 = dot_product(Hs,F)

        !8 - g(-eps,eps)

        Y1 = Yb-eps*H1+eps*H2

        call FEqsBfc(n,Y1,F,normF,iErr)

        g8 = dot_product(Hs,F)



        !a = <Hs,DuuFH1H1>

        !a = 1/eps^2 (g(eps,0)-2*g(0,0)+g(-eps,0))

        a = 1/eps**2*(g1-2*g0+g3)

        !c = <Hs,DuuFH2H2>

        !c = 1/eps^2 (g(0,eps)-2*g(0,0)+g(0,-eps))

        c = 1/eps**2*(g2-2*g0+g4)

        !b = <Hs,DuuFH1H2>

        !b = 1/4*1/eps^2 (g(eps,eps)+g(-eps,-eps)-g(eps,-eps)-g(-eps,eps))

        b = 0.25*1/eps**2*(g5+g6-g7-g8)



    endsubroutine  ApprxBifEqu
