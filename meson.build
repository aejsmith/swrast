project('swrast',
        'cpp',
        default_options: ['cpp_std=c++14'])

cpp_args = ['-Wall', '-Werror', '-Wno-unused-function', '-Wno-unused-variable']

foreach a : cpp_args
    add_project_arguments(a, language: 'cpp')
endforeach

sdl_dep = dependency('sdl2')

sources = [
    'src/CSurface.cpp',
    'src/main.cpp',
]

executable('swrast',
           dependencies: [sdl_dep],
           sources:      sources)
