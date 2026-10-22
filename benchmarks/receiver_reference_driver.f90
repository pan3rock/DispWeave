! Copyright 2019 Pascal Audet

! This file is part of Telewavesim.

! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:

! The above copyright notice and this permission notice shall be included in all
! copies or substantial portions of the Software.

! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
! SOFTWARE.

! Validation driver by DispWeave; free-surface assembly follows plane_land.
! DispWeave validation driver, calling unmodified Telewavesim 0.2.1 rmatrix.
! Uses real angular frequencies (plane_land otherwise adds 0.001*i damping).
program reference
use conf
use rmat
implicit none
integer :: nlay,nfreq,i,j,k,l,m,ii
real(8) :: p,f,vp,vs,h,dens,lam,mu0,px,py
complex(8) :: tus(3,3),rus(3,3),tds(3,3),rds(3,3),omega
complex(8) :: ndinv(3,3),rf(3,3),inverse(3,3),eye(3,3),surface(3,3),incident(3),u(3)
read(*,*) nlay,nfreq,p
if(nlay>nlaymx) stop 1
a=0;rho=0;thickn=0;isoflg=1
px=p/1000;py=0
 do m=1,nlay
  read(*,*) h,vs,vp,dens
  thickn(m)=h*1000;rho(m)=dens*1000
  mu0=(vs*1000)**2;lam=(vp*1000)**2-2*mu0
  do i=1,3
   do j=1,3
    do k=1,3
     do l=1,3
      if(i==j.and.k==l) a(i,j,k,l,m)=a(i,j,k,l,m)+lam
      if(i==k.and.j==l) a(i,j,k,l,m)=a(i,j,k,l,m)+mu0
      if(i==l.and.j==k) a(i,j,k,l,m)=a(i,j,k,l,m)+mu0
     enddo
    enddo
   enddo
  enddo
 enddo
 eye=0
 do i=1,3
  eye(i,i)=1
 enddo
 incident=0;incident(1)=1
 do ii=1,nfreq
  read(*,*) f
  omega=cmplx(2*pi*f,0d0,kind=8)
  call rmatrix(px,py,omega,nlay,tus,rus,tds,rds,ii-1)
  call xinv3(evecs(4:6,1:3,1),ndinv)
  rf=-matmul(ndinv,evecs(4:6,4:6,1))
  call xinv3(eye-matmul(rds,rf),inverse)
  surface=evecs(1:3,4:6,1)+matmul(evecs(1:3,1:3,1),rf)
  u=matmul(surface,matmul(inverse,matmul(tus,incident)))
  write(*,'(5ES26.17)') f,real(u(1)),aimag(u(1)),-real(u(3)),-aimag(u(3))
 enddo
end program
