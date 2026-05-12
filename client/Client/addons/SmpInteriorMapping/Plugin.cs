using Godot;

using System;
using System.Diagnostics.CodeAnalysis;

#if TOOLS

// ReSharper disable once CheckNamespace
namespace SmpInteriorMapping;

[Tool]
internal partial class Plugin : EditorPlugin
{
    [field: AllowNull, MaybeNull]
    private SmpImportPlugin SmpImportPlugin => field ??= new();

    public override void _EnterTree()
    {
        AddImportPlugin(SmpImportPlugin);
        GD.Print($"[{nameof(SmpInteriorMapping)}]: Loaded.");
    }

    public override void _ExitTree()
    {
        RemoveImportPlugin(SmpImportPlugin);
        GD.Print($"[{nameof(SmpInteriorMapping)}]: Unloaded.");
    }
}
#endif