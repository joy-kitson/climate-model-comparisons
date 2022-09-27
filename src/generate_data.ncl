; Calculate pairwise correlations, similarity score, weights and weighted relative errors
begin

nv  = 66  ; number of metrics
nr  = 4   ; number of regions
ng  = 37   ; number of GCMs
Dx  =0.3  ; Raidus of similarity

ss  = new((/nr,nv,nv/),"float")
rr  = new((/nr,nv/),"float")
cor = new((/nr,nv,nv/),"float")
weights = new((/nr,nv/),"float")
weighted = new((/nr,nv,ng/),"float")
unweighted = new((/nr,nv,ng/),"float")

fin = addfile("unweighted_data_37gcms_66metrics_0708.nc","r")

do r=0,nr-1 ; four conus regions

vin = fin->unweighted_data(r,:,:) ; (var, model)

do k =0,nv-1 ;metrics loop
do j = 0,nv-1

cor(r,j,k) = escorc(vin(j,:),vin(k,:))

end do
end do

cor@_FillValue=-999

tcor=1-abs(cor(r,:,:))
tcor = where(tcor.eq.0.,tcor@_FillValue,tcor)
cor_1d = abs(ndtooned(tcor))
ss(r,:,:) = exp(-1.*(tcor/Dx)^2)

do v = 0, nv-1
rr(r,v) = 1 + sum(ss(r,v,:))
end do

ww = 1/rr(r,:)

; rank models
weighted(r,:,:) = vin*conform(vin,ww,(/0/))
unweighted(r,:,:) = vin
weights(r,:) = ww
end do

unweighted!0="regions"
unweighted!1="metrics"
unweighted!2="gcms"

copy_VarCoords(unweighted,weighted)
unweighted@long_name ="unweighted relative error"
weighted@long_name   ="weighted relative error"

ss!0="regions"
ss!1="metrics"
ss!2="metrics"
copy_VarCoords(ss,cor)
ss@long_name ="Similarity Score"
cor@long_name="Pairwise correlations"

weights!0="regions"
weights!1="metrics"

cor@long_name="Pairwise correlations"

fout = addfile("data_cor_ss_ww.nc","c")

fout->unweighted_data=unweighted
fout->weighted_data=weighted
fout->ww=weights
fout->ss=ss
fout->cor=cor


end
