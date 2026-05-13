shader_type spatial;
render_mode cull_disabled, unshaded;

uniform sampler2D tex: repeat_disable, source_color;

group_uniforms Interior_Settings;
uniform float room_depth : hint_range(0.1, 10.0) = 1.0;
uniform vec2 back_wall_pos = vec2(0.5);
uniform vec2 back_wall_size = vec2(0.5);

instance uniform vec2 origin_size = vec2(1.0); // 期望的房间比例，例如 (2.0, 1.0)
instance uniform vec2 uv_scale = vec2(1);
instance uniform vec2 uv_offset = vec2(0);
instance uniform vec2 uv_margin = vec2(0);

varying vec3 view_dir;

void vertex() {
	UV *= vec2(1, -1);
	UV += uv_offset;
	UV *= uv_scale;
	
	vec3 cam_pos_model = (inverse(MODEL_MATRIX) * INV_VIEW_MATRIX * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
	vec3 view_dir_model = VERTEX - cam_pos_model;
	vec3 t = normalize(TANGENT);
	vec3 b = normalize(BINORMAL);
	vec3 n = normalize(NORMAL);
	
	view_dir = vec3(
		dot(view_dir_model, t),
		dot(view_dir_model, b),
		dot(view_dir_model, n)
	);
}

void fragment() {
	vec3 vd = normalize(view_dir);
	
	// 1. 将 UV 映射到以中心为原点的空间
	vec2 uv_unscaled = (fract(UV) - .5) / (1. - uv_margin) + .5;
	
	// 2. 核心：将射线原点 ro 映射到由 origin_size 定义的虚拟盒空间
	// 使用 origin_size 缩放 UV 空间，使房间在逻辑上是“立方体”
	vec3 ro = vec3((uv_unscaled - 0.5) * origin_size, 0.0);
	
	// 3. 射线方向根据 origin_size 调整步长
	// 这样确保了射线在长方形物体内部扫描时，X和Y的步进是均匀的
	vec3 rd = vec3(vd.x, vd.y, abs(vd.z) / room_depth);
	
	// 防除零
	rd.x = abs(rd.x) < 0.0001 ? 0.0001 * sign(rd.x) : rd.x;
	rd.y = abs(rd.y) < 0.0001 ? 0.0001 * sign(rd.y) : rd.y;
	rd.z = max(rd.z, 0.0001);
	
	// 4. 计算与虚拟盒子 (由 origin_size 限制范围) 的交点
	// 盒子边界是 [-origin_size/2, origin_size/2]
	vec2 half_size = origin_size * 0.5;
	float tx = ((step(0.0, rd.x) * origin_size.x - half_size.x) - ro.x) / rd.x;
	float ty = ((step(0.0, rd.y) * origin_size.y - half_size.y) - ro.y) / rd.y;
	float tz = (1.0 - ro.z) / rd.z; 
	
	float t = min(tz, min(tx, ty));
	vec3 hit = ro + rd * t;
	
	// 5. 将击中点从 [-origin_size/2, origin_size/2] 映射回 [0, 1] UV 空间
	vec2 hit_uv = (hit.xy / origin_size) + 0.5;
	
	// 6. 根据深度 Z 计算最终贴图采样区域
	vec2 back_min = back_wall_pos - back_wall_size * 0.5;
	vec2 back_max = back_wall_pos + back_wall_size * 0.5;
	
	vec2 uv_min = mix(vec2(0.0), back_min, hit.z);
	vec2 uv_max = mix(vec2(1.0), back_max, hit.z);
	
	// 使用 hit_uv 进行插值
	vec2 final_uv = mix(uv_min, uv_max, hit_uv) * vec2(1, -1) + vec2(0, 1);
	
	ALBEDO = texture(tex, final_uv).rgb;
}