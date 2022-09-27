; Sample script creat heatmaps. This example reads pairwise correlations
; and plots them for each region

; Command line args:
;   dir: the directory to load input data from and save output to
begin

region = (/"north","east","west","south"/)
pdata = new((/68,68/),"float")
f1=addfile(dir + "data_cor_ss_ww.nc","r")
gcms=asciiread(dir + "gcms_names.txt",-1,"string")
vars=asciiread(dir + "metrics_list.txt",-1,"string")

data=f1->cor(:,:,:)
var0=vars
vars=vars(::-1)
do r=0,3 ; loop over regions
  data0=data(r,:,:)

  do j =0,65 ;var loop
    do k = j+1,65
      data0(j,k) = -999
    end do
  end do

  data0@_FillValue=-999

  pdata=-999
  pdata!0="lat"
  pdata!1="time"
  pdata@_FillValue =-999
  pdata(1:66,1:66) = (/data0/)

  wks   = gsn_open_wks ("X11", "xxx_"+region(r) )  ; send graphics to PNG file

  res                  = True                ; plot mods desired
  res@gsnFrame = False
  resgsnMaximize =True
  res@pmTickMarkDisplayMode="NoCreate"
  res@cnFillOn         = True                ; color on
  res@cnLinesOn        = False               ; turn off contour lines
  res@cnFillPalette    = "testcmap"    ; set color map
  res@cnFillMode = "CellFill"
  res@cnLevelSelectionMode = "ManualLevels" ; manual levels
  res@cnMinLevelValF       = -1
  res@cnMaxLevelValF       =  1
  res@cnLevelSpacingF      =  0.01
  txres = True
  txres@txFontHeightF = 0.006
  txres@txFontQuality = "High"
  txres@txAngleF=90
  txres@txJust = "CenterLeft"
  tyres = True
  tyres@txFontHeightF = 0.006
  tyres@txFontQuality = "High"
  tyres@txJust = "CenterRight"
  res@lbBoxSeparatorLinesOn = False
  res@lbLabelFontHeightF =.011
  res@lbLabelStride         = 10
  lbres                      = True
  lbres@vpHeightF            = 0.4
  lbres@vpWidthF             = 0.01 
  res@vpHeightF =.7
  res@vpWidthF = .7
  res@vpYF =.87
  res@vpXF =.2
  plot = gsn_csm_lat_time(wks, pdata, res)

  a=0.2+0.010448
  b=.87-0.010448
  do m=0,65
    gsn_text_ndc(wks,vars(m),0.2,b,tyres)
    gsn_text_ndc(wks,vars(m),a,0.87,txres)
    b=b-0.010448
    a=a+0.010448
  end do

  frame(wks)
end do
end
