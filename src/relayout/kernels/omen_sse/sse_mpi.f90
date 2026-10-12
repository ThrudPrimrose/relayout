! Distributed SSE, same decomposition as sse_mpi.py: G energy-distributed in, Sigma momentum-distributed out.
! Column-major makes k the fastest index, so here the PACK is the strided gather and the unpack is contiguous
! per destination, the mirror of the C-order Python kernel.
! MPI symbols are external + parameter constants so the file lowers without mpi.mod (dace-fortran convention).
module sse_mpi_mod
  implicit none
contains
  subroutine sse_mpi(neigh_idx, dH, Gl, D, Sigmal, P, Nkzl, NEl, Nqz, Nw, N3D, NA, NB, Norb)
    integer, intent(in) :: P, Nkzl, NEl, Nqz, Nw, N3D, NA, NB, Norb
    integer, intent(in) :: neigh_idx(NA, NB)
    complex(8), intent(in) :: dH(NA, NB, N3D, Norb, Norb)
    complex(8), intent(in) :: Gl(P * Nkzl, NEl, NA, Norb, Norb)
    complex(8), intent(in) :: D(Nqz, Nw, NA, NB, N3D, N3D)
    complex(8), intent(inout) :: Sigmal(Nkzl, P * NEl, NA, Norb, Norb)
    complex(8) :: send(Nkzl, NEl, NA, Norb, Norb, P), recv(Nkzl, NEl, NA, Norb, Norb, P)
    complex(8) :: Gk(Nkzl, P * NEl, NA, Norb, Norb)
    complex(8) :: dHG(Norb, Norb), dHD(Norb, Norb)
    integer :: k, e, q, w, i, j, a, b, s, ierr, cnt
    integer, parameter :: MPI_COMM_WORLD = 0
    integer, parameter :: MPI_DOUBLE_COMPLEX = 22
    external :: MPI_Alltoall

    do s = 1, P
      send(:, :, :, :, :, s) = Gl((s - 1) * Nkzl + 1:s * Nkzl, :, :, :, :)
    end do
    cnt = Nkzl * NEl * NA * Norb * Norb
    call MPI_Alltoall(send, cnt, MPI_DOUBLE_COMPLEX, recv, cnt, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD, ierr)
    do s = 1, P
      Gk(:, (s - 1) * NEl + 1:s * NEl, :, :, :) = recv(:, :, :, :, :, s)
    end do

    do k = 1, Nkzl
      do e = 1, P * NEl
        do q = 1, Nqz
          do w = 0, Nw - 1
            do i = 1, N3D
              do j = 1, N3D
                do a = 1, NA
                  do b = 1, NB
                    if (e - w >= 1) then
                      dHG = matmul(Gk(k, e - w, neigh_idx(a, b) + 1, :, :), dH(a, b, i, :, :))
                      dHD = dH(a, b, j, :, :) * D(q, w + 1, a, b, i, j)
                      Sigmal(k, e, a, :, :) = Sigmal(k, e, a, :, :) + matmul(dHG, dHD)
                    end if
                  end do
                end do
              end do
            end do
          end do
        end do
      end do
    end do
  end subroutine sse_mpi
end module sse_mpi_mod
