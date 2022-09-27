; Calculate suitable value for raidus of similarity
begin
dir = "../data/"

nv  = 66  ; number of metrics
nit = 100 ; number of iterations
nr  = 4   ;number of regions

ros = new((/nr,nv,nit/),"float")
ss  = new((/nr,nv,nv/),"float")
rr  = new((/nr,nv/),"float")
cor = new((/nr,nv,nv/),"float")

fin = addfile(dir + "unweighted_data_37gcms_66metrics_0708.nc","r")

do z = 0, nit-1 ; change the radius from 0 to 1 in the increments of 0.01

  nn1=0.01+z/100.
  do r=0,nr-1 ; four conus regions

    vin = fin->unweighted_data(r,:,:) ; (var, model)
    do k =0,nv-1 ;metrics loop
      do j = 0,nv-1
        cor(r,j,k) = escorc(vin(j,:),vin(k,:))
      end do
    end do

    cor@_FillValue=-999
    opt = True
    opt@PrintStat = True
    tcor=1-abs(cor(r,:,:))
    tcor = where(tcor.eq.0.,tcor@_FillValue,tcor)
    cor_1d = abs(ndtooned(tcor))
    ss(r,:,:) = exp(-1.*(tcor/nn1)^2)

    do v = 0, nv-1
      rr(r,v) = 1 + sum(ss(r,v,:))
    end do

    ww = 1/rr(r,:)

    ros(r,:,z) = ww
  end do
end do

ros!0="regions"
ros!1="metrics"
ros!2="dx"
ros@long_name="radius of similarity (Dx)"
ros@_FillValue =-999

fout=addfile(dir + "ros_100.nc","c")
fout->ros=ros
end
