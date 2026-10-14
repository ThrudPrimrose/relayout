! SSE as written in OMEN. Natural index order; arrays are column-major, so the fastest index is k.
module sse_mod
  implicit none
contains
  subroutine sse(neigh_idx, dH, G, D, Sigma, Nkz, NE, Nqz, Nw, N3D, NA, NB, Norb)
    integer, intent(in) :: Nkz, NE, Nqz, Nw, N3D, NA, NB, Norb
    integer, intent(in) :: neigh_idx(NA, NB)
    complex(8), intent(in) :: dH(NA, NB, N3D, Norb, Norb)
    complex(8), intent(in) :: G(Nkz, NE, NA, Norb, Norb)
    complex(8), intent(in) :: D(Nqz, Nw, NA, NB, N3D, N3D)
    complex(8), intent(inout) :: Sigma(Nkz, NE, NA, Norb, Norb)
    complex(8) :: dHG(Norb, Norb), dHD(Norb, Norb)
    integer :: k, e, q, w, i, j, a, b

    do k = 1, Nkz
      do e = 1, NE
        do q = 1, Nqz
          do w = 0, Nw - 1
            do i = 1, N3D
              do j = 1, N3D
                do a = 1, NA
                  do b = 1, NB
                    if (e - w >= 1) then
                      ! neigh_idx holds 0-based atom ids, shared with the Python kernel.
                      dHG = matmul(G(k, e - w, neigh_idx(a, b) + 1, :, :), dH(a, b, i, :, :))
                      dHD = dH(a, b, j, :, :) * D(q, w + 1, a, b, i, j)
                      Sigma(k, e, a, :, :) = Sigma(k, e, a, :, :) + matmul(dHG, dHD)
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      end do
    end do
  end subroutine sse
end module sse_mod
