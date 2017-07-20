  SELECT cls.cls_id,obj.obj_id,upper(trim(replace(replace(replace(replace(replace(obj.name,chr(10),''),',',' '),chr(13),''),chr(9),' '),'"',''))) as name,obj.prj_id, 0 as pr
    FROM cs_art.obj,cs_art.ocl,cs_art.cls
    WHERE obj.mlt_id=1 AND obj.status=1
      AND ocl.mlt_id=obj.mlt_id AND ocl.clf_id=6 AND ocl.obj_id=obj.obj_id
      AND cls.mlt_id=ocl.mlt_id AND cls.clf_id=ocl.clf_id AND cls.cls_id=ocl.cls_id 
      AND cls.parent=0 -- Класс нижнего уровня
      AND cls.code NOT LIKE '00%' -- Исключая "технологические" классы
      AND cls.status<>2
      AND NOT EXISTS (SELECT 1 FROM cs_art.vobj
                        WHERE vobj.mlt_id=obj.mlt_id
                          AND vobj.obj_id=obj.obj_id
                          AND vobj.aobj_id=9135)
  union                         
  SELECT cls.cls_id, obj.obj_id,upper(trim(replace(replace(replace(replace(replace(oclp.sname,chr(10),''),',',' '),chr(13),''),chr(9),' '),'"',''))) as name,obj.prj_id, 1 as pr
    FROM cs_art.obj,cs_art.ocl,cs_art.cls,cs_art.oclp
    WHERE obj.mlt_id=1 AND obj.status=1
      AND ocl.mlt_id=obj.mlt_id AND ocl.clf_id=6 AND ocl.obj_id=obj.obj_id
      AND cls.mlt_id=ocl.mlt_id AND cls.clf_id=ocl.clf_id AND cls.cls_id=ocl.cls_id 
      AND cls.parent=0 -- Класс нижнего уровня
      AND cls.code NOT LIKE '00%' -- Исключая "технологические" классы
      AND cls.status<>2
      and oclp.obj_id=obj.obj_id
      and oclp.mlt_id=obj.mlt_id
      and oclp.clf_id=ocl.clf_id
      and oclp.cls_id=ocl.cls_id
      and oclp.prj_id=obj.prj_id
      AND NOT EXISTS (SELECT 1 FROM cs_art.vobj
                        WHERE vobj.mlt_id=obj.mlt_id
                          AND vobj.obj_id=obj.obj_id
                          AND vobj.aobj_id=9135)
                          
select * from aobj where aobj_id=9135                          