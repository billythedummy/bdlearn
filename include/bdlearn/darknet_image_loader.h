#ifdef __cplusplus  
extern "C" { 
#endif 
#ifndef _DARKNET_IMAGE_LOADER_H_
#define _DARKNET_IMAGE_LOADER_H_

#include <cstdio>

// CODE COPIED FROM darknet/uwimg
// for use under the 
// DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
// All credit to https://github.com/pjreddie

typedef struct dn_node{
    void *val;
    struct dn_node *next;
    struct dn_node *prev;
} dn_node;

typedef struct dn_list{
    int size;
    dn_node *front;
    dn_node *back;
} dn_list;

typedef struct{
    int w,h,c;
    float *data;
} dn_image;

dn_image make_empty_image(int w, int h, int c);

dn_image make_image(int w, int h, int c);

dn_image load_image_stb(char *filename, int channels);

dn_image load_image(char *filename);

dn_list *make_list();

void list_insert(dn_list *l, void *val);

char *fgetl(FILE *fp);

dn_list *get_lines(char const * filename);

void **list_to_array(dn_list *l);

void free_node(dn_node *n);

void free_list(dn_list *l);

#endif

#ifdef __cplusplus 
} 
#endif 