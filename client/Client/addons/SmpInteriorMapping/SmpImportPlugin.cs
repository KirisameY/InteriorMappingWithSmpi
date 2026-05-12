using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.Json;

using Godot;

using JetBrains.Annotations;

using FileAccess = Godot.FileAccess;

#if TOOLS

// ReSharper disable once CheckNamespace
namespace SmpInteriorMapping;

internal partial class SmpImportPlugin : EditorImportPlugin
{
    public override string _GetImporterName() => "kirisamey.smpimport.smp";

    public override string _GetVisibleName() => "S-MPI Interior Mapping";

    public override string[] _GetRecognizedExtensions() => ["smp"];

    public override string _GetSaveExtension() => "res";

    public override string _GetResourceType() => nameof(ShaderMaterial);

    public override int _GetPresetCount() => 1;

    public override string _GetPresetName(int presetIndex) => "Default";

    public override Godot.Collections.Array<Godot.Collections.Dictionary> _GetImportOptions(string path, int presetIndex) => [];

    public override Error _Import(string sourceFile, string savePath,
                                  Godot.Collections.Dictionary options,
                                  Godot.Collections.Array<string> platformVariants,
                                  Godot.Collections.Array<string> genFiles)
    {
        using var file = FileAccess.Open(sourceFile, FileAccess.ModeFlags.Read);
        if (file.GetError() != Error.Ok) return Error.Failed;

        var fileContent = file.GetBuffer((long)file.GetLength());
        using var fileBuf = new MemoryStream(fileContent);
        using var zip = new ZipArchive(fileBuf);
        using var plansJson = zip.GetEntry("planes.json")?.Open();
        using var viewportJson = zip.GetEntry("viewport.json")?.Open();
        //var texPng = zip.GetEntry("textures.png")?.Open();

        if (plansJson is null || viewportJson is null) return Error.Failed;

        var jsonOpt = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };

        var planeData = JsonSerializer.Deserialize<List<PlaneData>>(plansJson, jsonOpt);
        if (planeData is null) return Error.Failed;
        var planeCount = planeData.Count;
        var shaderStr = GenerateShader(planeData);
        var shader = new Shader();
        shader.Code = shaderStr;

        var texPngs = Enumerable.Range(0, planeCount)
                                .Select(i => zip.GetEntry($"planes/{i}.png"))
                                .Select(e => e?.Open())
                                .ToArray();
        if (texPngs.Any(e => e is null)) return Error.Failed;
        var texs = new Texture2DArray();
        texs.CreateFromImages(new Godot.Collections.Array<Image>(texPngs.Select(s =>
        {
            var img = new Image();
            using var buf = new MemoryStream();
            s!.CopyTo(buf);
            img.LoadPngFromBuffer(buf.GetBuffer());
            return img;
        })));

        var viewportData = JsonSerializer.Deserialize<ViewportData>(viewportJson, jsonOpt);

        var mat = new ShaderMaterial();
        mat.Shader = shader;
        mat.SetShaderParameter("texs", texs);
        mat.SetShaderParameter("origin_size", new Vector2((float)viewportData.Width, (float)viewportData.Height));

        string filename = $"{savePath}.{_GetSaveExtension()}";
        return ResourceSaver.Save(mat, filename);
    }

    private static String GenerateShader(IReadOnlyList<PlaneData> planes)
    {
        var builder = new StringBuilder();

        builder.AppendLine("shader_type spatial;")
               .AppendLine("render_mode cull_disabled, unshaded;")
               .AppendLine()
               .AppendLine("uniform sampler2DArray texs: repeat_disable;")
               .AppendLine("uniform vec2 origin_size = vec2(1);")
               .AppendLine()
               .AppendLine("instance uniform vec2 uv_scale = vec2(1);")
               .AppendLine("instance uniform vec2 uv_offset = vec2(0);")
               .AppendLine("instance uniform vec2 uv_margin = vec2(0);")
               .AppendLine()
               .AppendLine("varying vec3 view_dir;")
               .AppendLine();


        var formatVec = (List<double> v) => $"vec3({v[0]}, {v[1]}, {v[2]})";
        var vecs = new (string, Func<PlaneData, List<double>>)[]
        {
            ("o", p => p.Center),
            ("n", p => p.Normal),
            ("u", p => p.LengthVec),
            ("v", p => p.WidthVec)
        };
        foreach (var (s, vecGet) in vecs)
        {
            builder.AppendLine($"const vec3[{planes.Count}] planes_{s} = {{")
                   .AppendLine("    " + string.Join(",\n    ", planes.Select(vecGet).Select(formatVec)))
                   .AppendLine("};");
        }

        builder.AppendLine("""
            void vertex() {
                // Called for every vertex the material is visible on.
                UV *= vec2(1, -1);
                UV += uv_offset;
                UV *= uv_scale;

                // 切线空间变换
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
            
            """
        );

        var n = planes.Count;
        builder.AppendLine(
            $$"""
            void fragment() {
                // Called for every pixel the material is visible on.
                
                vec3 vd = normalize(view_dir);
                vec2 uv_unscaled = (fract(UV) - .5) / (1. - uv_margin) + .5;
                vec3 org = vec3(uv_unscaled * origin_size - origin_size/2., 0);
                
                vec4[{{n}}] infos;
            """
        );

        builder.AppendLine()
               .AppendLine("    // 初始化数组");

        for (int i = 0; i < n; i++)
        {
            builder.AppendLine($"    infos[{i}] = vec4(-99999.);");
        }

        builder.AppendLine()
               .AppendLine("    // 遍历平面求交点信息并排序插入");

        for (int i = 0; i < n; i++)
        {
            builder.AppendLine($"    // {i}")
                   .AppendLine("    {")
                    // t = (P - O) · N / (D · N)
                   .AppendLine($"""
                                float t = dot(planes_o[{i}] - org, planes_n[{i}]) / dot(vd, planes_n[{i}]);
                                vec3 pos = org + vd * t;
                                vec3 pos_p = pos - planes_o[{i}];
                                float u = dot(pos_p, planes_u[{i}]) / dot(planes_u[{i}], planes_u[{i}]);
                                float v = dot(pos_p, planes_v[{i}]) / dot(planes_v[{i}], planes_v[{i}]);
                        """
                    );
            builder.AppendLine($"        vec4 current = vec4(t, float({i}), u, v);");
            for (int j = 0; j <= i; j++)
            {
                builder.AppendLine($"    // {i} - {j}")
                       .AppendLine("        {")
                       .AppendLine($"""
                                        float swap = step(infos[{j}].x, current.x);
                                        vec4 tmp = infos[{j}];
                                        infos[{j}] = infos[{j}] * (1. - swap) + current * swap;
                                        current = tmp * swap + current * (1. - swap);
                            """
                        )
                       .AppendLine("        }");
            }
            builder.AppendLine("    }");
        }

        builder.AppendLine()
               .AppendLine("    // 遍历排序信息，采样并alpha混合")
               .AppendLine("    vec3 final_color = vec3(0);");

        for (int j = 0; j < n; j++)
        {
            builder.AppendLine($"    // {j}")
                   .AppendLine("    {");
            builder.AppendLine($"""
                        float d = infos[{j}].x;
                        float i = infos[{j}].y;
                        vec2 uv = infos[{j}].zw;
                        vec4 color = texture(texs, vec3(uv + vec2(.5), i));
                        color *= step(abs(uv.x), 0.5) * step(abs(uv.y), 0.5) * step(0, infos[{j}].x);
                        final_color *= 1.-color.a;
                        final_color += color.rgb * color.a;
                """
            );
            builder.AppendLine("    }");
        }

        builder.AppendLine("""
                
                ALBEDO = final_color;
            }
            """
        );

        return builder.ToString();
    }


    [UsedImplicitly]
    private struct PlaneData
    {
        public List<double> Center { get; set; }
        public List<double> Normal { get; set; }
        public List<double> LengthVec { get; set; }
        public List<double> WidthVec { get; set; }
    }

    [UsedImplicitly]
    private struct ViewportData
    {
        public double Width { get; set; }
        public double Height { get; set; }
    }
}

#endif