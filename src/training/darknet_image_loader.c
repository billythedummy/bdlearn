#include "bdlearn/darknet_image_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

dn_image make_empty_image(int w, int h, int c)
{
    dn_image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

dn_image make_image(int w, int h, int c)
{
    dn_image out = make_empty_image(w,h,c);
    out.data = (float *)calloc(h*w*c, sizeof(float));
    return out;
}

dn_image load_image_stb(char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n",
            filename, stbi_failure_reason());
        exit(0);
    }
    if (channels) c = channels;
    int i,j,k;
    dn_image im = make_image(w, h, c);
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index]/255.;
            }
        }
    }
    //We don't like alpha channels, #YOLO
    if(im.c == 4) im.c = 3;
    free(data);
    return im;
}

dn_image load_image(char *filename)
{
    dn_image out = load_image_stb(filename, 0);
    return out;
}

dn_list *make_list()
{
	dn_list *l = (dn_list*)malloc(sizeof(dn_list));
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
}

void list_insert(dn_list *l, void *val)
{
	dn_node *new_node = (dn_node*)malloc(sizeof(dn_node));
	new_node->val = val;
	new_node->next = 0;

	if(!l->back){
		l->front = new_node;
		new_node->prev = 0;
	}else{
		l->back->next = new_node;
		new_node->prev = l->back;
	}
	l->back = new_node;
	++l->size;
}

char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    size_t size = 512;
    char *line = (char*)malloc(size*sizeof(char));
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while((line[curr-1] != '\n') && !feof(fp)){
        if(curr == size-1){
            size *= 2;
            line = (char*)realloc(line, size*sizeof(char));
            if(!line) {
                fprintf(stderr, "malloc failed %ld\n", size);
                exit(0);
            }
        }
        size_t readsize = size-curr;
        if(readsize > INT_MAX) readsize = INT_MAX-1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if(line[curr-1] == '\n') line[curr-1] = '\0';
    return line;
}

dn_list *get_lines(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) {
        fprintf(stderr, "Couldn't open file %s\n", filename);
        exit(0);
    }
    dn_list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

void **list_to_array(dn_list *l)
{
    void **a = (void **)calloc(l->size, sizeof(void*));
    int count = 0;
    dn_node *n = l->front;
    while(n){
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}


void free_node(dn_node *n)
{
	dn_node *next;
	while(n) {
		next = n->next;
		free(n);
		n = next;
	}
}

void free_list(dn_list *l)
{
	free_node(l->front);
	free(l);
}